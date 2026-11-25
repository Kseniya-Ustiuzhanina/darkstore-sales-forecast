"""
DAG батч-инференса прогноза недельных продаж «Прилавка».

1. load_data — история продаж и план из PostgreSQL (таблицы sales, plan, stores, features).
2. preprocess_features — признаки той же функцией preprocess_data, что при обучении.
3. load_model — модель CatBoost из S3 (Yandex Object Storage).
4. run_inference — прогноз, отрицательные значения заменяются на 0.
5. save_predictions — запись в таблицу predictions.

Креды не хранятся в коде: S3 — через Airflow Variables
(s3_bucket_name, s3_model_key, s3_access_key, s3_secret_key),
PostgreSQL — через Airflow Connection postgres_sales_db.
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.models import Variable
from airflow.hooks.postgres_hook import PostgresHook
import pandas as pd
import numpy as np
import boto3
from catboost import CatBoostRegressor
import logging
import io
import pickle

# ========== Функция предобработки данных ==========
from preprocessing import preprocess_data
from psycopg2.extras import execute_values

# ========== Конфигурация ==========

S3_BUCKET = Variable.get("s3_bucket_name", default_var="your-bucket-name")
S3_ACCESS_KEY = Variable.get("s3_access_key", default_var=None)
S3_SECRET_KEY = Variable.get("s3_secret_key", default_var=None)
 
S3_MODEL_KEY = Variable.get("s3_model_key", default_var="catboost_model.pkl")
POSTGRES_CONN_ID = "postgres_sales_db"  # ID подключения в Airflow Connections
 
RANDOM_STATE = 42
CAT_FEATURES = ['store', 'dept', 'is_holiday', 'type']
 
S3_ENDPOINT = 'https://storage.yandexcloud.net'
MODEL_TMP_PATH = '/tmp/catboost_sales_model.pkl'
 
# Признаки модели строго в порядке обучения (model_training.ipynb)
FEATURE_COLS = ['store', 'dept', 'type', 'size', 'is_holiday',
                'avg_sales_before', 'median_sales_before',
                'sales_1week_ago', 'sales_2week_ago', 'sales_4week_ago', 'sales_52week_ago',
                'mean_sales_2week', 'mean_sales_4week']
 
# Краткосрочные признаки истории продаж: на горизонте плана фактических продаж нет,
# поэтому по условию задания пропуски в них заполняются нулями (учебное упрощение).
# Лаг 52 недели на весь горизонт плана известен из истории и нулями не заполняется;
# пропуск в нём (пара без годовой истории) CatBoost обрабатывает сам.
YEARLY_LAG = 'sales_52week_ago'
LAG_COLS = ['sales_1week_ago', 'sales_2week_ago', 'sales_4week_ago',
            'mean_sales_2week', 'mean_sales_4week']
 
logger = logging.getLogger(__name__)
 
 
# ========== Задачи DAG ==========
 
def load_data_from_postgres(**context):
    """
    Собирает историю продаж (до первой даты плана) и план (weekly_sales = NULL)
    в таблицу inference_data_temp и присоединяет stores и features.
    Первую дату плана передаёт в XCom.
    """
    hook = PostgresHook(postgres_conn_id=POSTGRES_CONN_ID)
 
    plan_start = hook.get_first('SELECT MIN(date) FROM plan')[0]
 
    hook.run(["""
        DROP TABLE IF EXISTS inference_data_temp
    """, f"""
        CREATE TABLE inference_data_temp AS
        WITH combined AS (
            SELECT store, dept, date, weekly_sales, is_holiday
            FROM sales
            WHERE date < '{plan_start}'
            UNION ALL
            SELECT store, dept, date, NULL::double precision AS weekly_sales, is_holiday
            FROM plan
        )
        SELECT c.*, st.type, st.size,
               f.temperature, f.fuel_price,
               f.factor1, f.factor2, f.factor3, f.factor4, f.factor5,
               f.cpi, f.unemployment
        FROM combined c
        LEFT JOIN stores st ON st.store = c.store
        LEFT JOIN features f
               ON f.store = c.store AND f.dept = c.dept AND f.date = c.date
    """])
 
    n_rows, d_min, d_max = hook.get_first(
        'SELECT COUNT(*), MIN(date), MAX(date) FROM inference_data_temp')
    logger.info('inference_data_temp: %d строк, %s — %s', n_rows, d_min, d_max)
 
    context['ti'].xcom_push(key='plan_start_date', value=str(plan_start))
 
 
def preprocess_features(**context):
    """
    Считает признаки на склеенных данных, оставляет только строки плана,
    заполняет краткосрочные лаги нулями, удаляет строки с NaN (кроме лага 52 недели)
    и передаёт датафрейм в XCom.
    Временную таблицу inference_data_temp удаляет.
    """
    plan_start = context['ti'].xcom_pull(key='plan_start_date', task_ids='load_data')
    hook = PostgresHook(postgres_conn_id=POSTGRES_CONN_ID)
 
    df = hook.get_pandas_df('SELECT * FROM inference_data_temp')
    logger.info('Загружено из inference_data_temp: %d строк', len(df))
 
    df = preprocess_data(df)
 
    df = df[df['date'] >= pd.Timestamp(plan_start)].drop(columns=['weekly_sales'])
    df[LAG_COLS] = df[LAG_COLS].fillna(0)
 
    df = df[['store', 'dept', 'date'] + FEATURE_COLS[2:]]
    n_before = len(df)
    df = df.dropna(subset=[c for c in df.columns if c != YEARLY_LAG]).reset_index(drop=True)
    logger.info('Строк плана: %d, удалено с NaN: %d, осталось: %d',
                n_before, n_before - len(df), len(df))
 
    context['ti'].xcom_push(key='inference_data',
                            value=df.to_json(orient='split', date_format='iso'))
 
    hook.run('DROP TABLE IF EXISTS inference_data_temp')
 
 
def load_model_from_s3(**context):
    """
    Загружает модель CatBoost из S3 (Yandex Cloud) в память,
    проверяет, что это модель, и сохраняет во временный файл.
    В XCom передаёт путь к файлу.
    """
    s3 = boto3.client(
        's3',
        endpoint_url=S3_ENDPOINT,
        aws_access_key_id=S3_ACCESS_KEY,
        aws_secret_access_key=S3_SECRET_KEY,
    )
 
    buffer = io.BytesIO()
    s3.download_fileobj(S3_BUCKET, S3_MODEL_KEY, buffer)
    buffer.seek(0)
    model = pickle.load(buffer)
    logger.info('Модель загружена из s3://%s/%s: %s, признаков: %d',
                S3_BUCKET, S3_MODEL_KEY, type(model).__name__, len(model.feature_names_))
 
    with open(MODEL_TMP_PATH, 'wb') as f:
        pickle.dump(model, f)
 
    context['ti'].xcom_push(key='model_path', value=MODEL_TMP_PATH)
 
 
def run_batch_inference(**context):
    """
    Применяет модель к подготовленным данным.
    Отрицательные и нечисловые предсказания заменяет на 0.
    Передаёт в XCom датафрейм с предсказаниями.
    """
    data_json = context['ti'].xcom_pull(key='inference_data', task_ids='preprocess_features')
    model_path = context['ti'].xcom_pull(key='model_path', task_ids='load_model')
 
    df = pd.read_json(data_json, orient='split')
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
 
    if list(model.feature_names_) != FEATURE_COLS:
        raise ValueError(f'Признаки модели {model.feature_names_} не совпадают с FEATURE_COLS')
 
    df['predicted_weekly_sales'] = model.predict(df[FEATURE_COLS])
 
    bad = ~np.isfinite(df['predicted_weekly_sales']) | (df['predicted_weekly_sales'] < 0)
    df.loc[bad, 'predicted_weekly_sales'] = 0
    logger.info('Предсказаний: %d, заменено на 0: %d', len(df), bad.sum())
 
    result = df[['store', 'dept', 'date', 'predicted_weekly_sales']]
    context['ti'].xcom_push(key='predictions',
                            value=result.to_json(orient='split', date_format='iso'))
 
 
def save_predictions_to_postgres(**context):
    """
    Пересоздаёт таблицу predictions и записывает в неё предсказания.
    """
    data_json = context['ti'].xcom_pull(key='predictions', task_ids='run_inference')
    df = pd.read_json(data_json, orient='split')
 
    hook = PostgresHook(postgres_conn_id=POSTGRES_CONN_ID)
    conn = hook.get_conn()
    cur = conn.cursor()
 
    cur.execute('DROP TABLE IF EXISTS predictions')
    cur.execute("""
        CREATE TABLE predictions (
            store INT,
            dept INT,
            date DATE,
            predicted_weekly_sales FLOAT,
            prediction_timestamp TIMESTAMP
        )
    """)
 
    ts = datetime.now()
    rows = [(int(r.store), int(r.dept), r.date.date(), float(r.predicted_weekly_sales), ts)
            for r in df.itertuples(index=False)]
    execute_values(cur,
                   'INSERT INTO predictions '
                   '(store, dept, date, predicted_weekly_sales, prediction_timestamp) VALUES %s',
                   rows, page_size=10000)
    conn.commit()
 
    cur.execute('SELECT COUNT(*) FROM predictions')
    logger.info('Строк в predictions: %d', cur.fetchone()[0])
    logger.info('Первые 5 строк:\n%s', hook.get_pandas_df(
        'SELECT * FROM predictions ORDER BY store, dept, date LIMIT 5').to_string())
 
    cur.close()
    conn.close()
 
 
# ========== Определение DAG ==========
 
default_args = {
    'owner': 'Устюжанина',
    'depends_on_past': False,
    'email': [],
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=1),
}
 
dag = DAG(
    'sales_prediction_batch_inference',
    default_args=default_args,
    description='Batch-инференс прогнозирования продаж для Прилавка',
    schedule_interval='0 20 * * 0',  # Каждое воскресенье в 20:00
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=['sales', 'ml', 'batch-inference', 'production'],
)
 
task_load_data = PythonOperator(
    task_id='load_data',
    python_callable=load_data_from_postgres,
    provide_context=True,
    dag=dag,
)
 
task_preprocess = PythonOperator(
    task_id='preprocess_features',
    python_callable=preprocess_features,
    provide_context=True,
    dag=dag,
)
 
task_load_model = PythonOperator(
    task_id='load_model',
    python_callable=load_model_from_s3,
    provide_context=True,
    dag=dag,
)
 
task_inference = PythonOperator(
    task_id='run_inference',
    python_callable=run_batch_inference,
    provide_context=True,
    dag=dag,
)
 
task_save_predictions = PythonOperator(
    task_id='save_predictions',
    python_callable=save_predictions_to_postgres,
    provide_context=True,
    dag=dag,
)
 
task_load_data >> task_preprocess
task_load_model >> task_inference
task_preprocess >> task_inference >> task_save_predictions