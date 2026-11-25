"""
Предобработка и признаки для модели продаж «Прилавка».
Те же функции, что в model_training.ipynb, — обучение и инференс считают признаки одинаково.
"""

import pandas as pd
import numpy as np

import logging
logger = logging.getLogger(__name__)

TARGET = 'weekly_sales'
SERVICE_COLS = ['store', 'dept', 'is_holiday', TARGET]


def create_temporal_features(df):
    """Календарные признаки из даты недели (date — воскресенье)."""
    df = df.copy()
    df['month'] = df['date'].dt.month
    df['quarter'] = df['date'].dt.quarter
    df['year'] = df['date'].dt.year
    df['week'] = df['date'].dt.isocalendar().week.astype(int)
    return df


def create_avg_sales_feature(df):
    """Средняя и медиана продаж пары «даркстор–отдел» по всем неделям строго до текущей."""
    df = df.sort_values(['store', 'dept', 'date']).reset_index(drop=True)
    past = df.groupby(['store', 'dept'])[TARGET].shift(1)
    grp = past.groupby([df['store'], df['dept']])
    df['avg_sales_before'] = grp.transform(lambda s: s.expanding().mean())
    df['median_sales_before'] = grp.transform(lambda s: s.expanding().median())
    return df


def create_lag_features(df):
    """Продажи той же пары ровно 1, 2, 4 и 52 недели назад — по дате, а не по позиции строки."""
    df = df.copy()
    base = df[['store', 'dept', 'date', TARGET]]
    for k in (1, 2, 4, 52):
        lag = (base.assign(date=base['date'] + pd.Timedelta(weeks=k))
                   .rename(columns={TARGET: f'sales_{k}week_ago'}))
        df = df.merge(lag, on=['store', 'dept', 'date'], how='left')
    return df


def create_rolling_features(df):
    """Среднее продаж пары за 2 и 4 календарные недели строго до текущей."""
    df = df.sort_values(['store', 'dept', 'date']).reset_index(drop=True)
    for w in (2, 4):
        rolled = (df.groupby(['store', 'dept'])
                    .rolling(f'{7 * w}D', on='date', closed='left')[TARGET]
                    .mean())
        assert (rolled.index.get_level_values(0).to_numpy() == df['store'].to_numpy()).all()
        assert (rolled.index.get_level_values(1).to_numpy() == df['dept'].to_numpy()).all()
        df[f'mean_sales_{w}week'] = rolled.to_numpy()
    return df


def preprocess_data(df):
    """
    Полная предобработка данных (идентична логике обучения).
    1. Дубликаты удаляются, отрицательные продажи (возвраты) обнуляются.
    2. Пропуски во внешних факторах заполняются средним.
       Пустой weekly_sales у строк плана НЕ заполняется: это неизвестное будущее.
    3. Строятся признаки: временные, средние, лаги, скользящие средние.
    """
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.drop_duplicates()

    df[TARGET] = df[TARGET].clip(lower=0)
    df['is_holiday'] = df['is_holiday'].astype(int)

    num_cols = df.select_dtypes(include='number').columns.difference(SERVICE_COLS)
    df[num_cols] = df[num_cols].fillna(df[num_cols].mean())

    df = create_temporal_features(df)
    df = create_avg_sales_feature(df)
    df = create_lag_features(df)
    df = create_rolling_features(df)

    logger.info('preprocess_data: %d строк, %s — %s',
                len(df), df['date'].min().date(), df['date'].max().date())
    return df
