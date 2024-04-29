import holidays
import pandas as pd


def data_preprocessing(
    df: pd.DataFrame,
    date: str,
    target_name: str,
    used_features: list,
    fill_target: bool = False,
):
    df[date] = pd.to_datetime(df[date], format="%Y-%m-%dT%H:%M:%SZ")

    start_date = df[date].min()
    end_date = df[date].max()
    all_dates = pd.date_range(start=start_date, end=end_date, freq="H")
    df_dates = pd.DataFrame(index=all_dates)
    df = df_dates.merge(df, how="inner", left_index=True, right_on=date)
    df = df.sort_values(by=date)
    date_col = df[date]
    df = df.drop(columns=[date])
    df = df.interpolate("akima", limit_direction="both", axis=0)
    df[date] = date_col
    df = df.fillna(df.median())
    df["month"] = df[date].dt.month
    df["day"] = df[date].dt.day
    df["hour"] = df[date].dt.hour
    df["day_of_week"] = df[date].dt.dayofweek

    ru_holidays = holidays.RU()
    df["holidays"] = df[date].transform(lambda x: int(x.replace(hour=0) in ru_holidays))

    if fill_target:
        df[target_name] = 0

    d = ["month", "day", "day_of_week", "hour"]
    d.append("holidays")
    d.extend(used_features)
    d.append(target_name)
    return df[d]
