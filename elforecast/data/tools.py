from typing import List

import holidays
import numpy as np
import pandas as pd
from pandas import DataFrame


def data_preprocessing(
    df: pd.DataFrame,
    date: str,
    target_name: str,
    used_features: list,
    max_miss: int = None,
    min_true: int = None,
    fill_target: bool = False,
    fill_date: bool = True,
    drop_if_less: bool = False,
    dl: dict = None,
    interpolate: dict = True,
) -> List[DataFrame]:
    df[date] = pd.to_datetime(df[date], format="%Y-%m-%dT%H:%M:%SZ")
    df.set_index(date, inplace=True)
    start = df.index.min()
    end = df.index.max()
    if not fill_target:
        df = df[df[target_name] != 0]
    if drop_if_less:
        for k in dl.keys():
            if k == target_name and fill_target:
                continue
            df = df.drop(df[df[k] < dl[k]].index)
    if not fill_target:
        df = df.dropna(subset=target_name)
    if fill_date:
        all_dates = pd.date_range(start=start, end=end, freq="H")
        df = df.reindex(all_dates)

    df["month"] = df.index.month
    df["day"] = df.index.day
    df["hour"] = df.index.hour
    df["day_of_week"] = df.index.dayofweek

    ru_holidays = holidays.RU()
    df["holidays"] = df.index.to_series().apply(lambda x: x.normalize() in ru_holidays)

    d = ["holidays"]
    for col, st, freq in zip(
        ["hour", "day", "month", "day_of_week"], [0, 1, 1, 0], [23, 30, 12, 6]
    ):
        df[f"sin_{col}"] = df[col].transform(
            lambda x, st=st, freq=freq: np.sin(2 * np.pi * (x - st) / freq)
        )
        df[f"cos_{col}"] = df[col].transform(
            lambda x, st=st, freq=freq: np.cos(2 * np.pi * (x - st) / freq)
        )
        d.extend([f"sin_{col}", f"cos_{col}"])
        d.append(col)
    d.extend(used_features)
    d.append(target_name)
    if fill_target:
        df[target_name] = 0

    df = df[d]

    if not interpolate:
        return [df.dropna()]
    if max_miss is None and min_true is None:
        return [df.interpolate(method="linear")]
    df_list = remove_miss(df, max_miss, min_true)

    for i in range(len(df_list)):
        df_list[i] = df_list[i].interpolate(method="linear", limit_direction="both")
    return df_list


def remove_miss(df: DataFrame, max_miss: int, min_true: int) -> List[DataFrame]:
    result = []
    nmiss = np.zeros(df.shape[1])
    start_pos = 0
    for i in range(df.shape[0]):
        nmiss += df.iloc[i].isna()
        if (nmiss > max_miss).any():
            nmiss = np.zeros(df.shape[1])
            if i - start_pos > min_true + max_miss:
                result.append(df[start_pos:i])
            start_pos = i + 1
    return result
