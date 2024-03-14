import numpy as np
import pandas as pd

MIN_BAROPRESSURE = 950

def data_preprocessing(df):

    df = df.drop(df[df['baropressure'] < MIN_BAROPRESSURE].index)
    df = df.dropna()

    df['Date'] = pd.to_datetime(df['Date'])

    start_date = df['Date'].min()
    end_date = df['Date'].max()
    all_dates = pd.date_range(start=start_date, end=end_date)
    df_dates = pd.DataFrame(index=all_dates)

    df = df_dates.merge(df, how='outer', left_index=True, right_on='Date')
    df = df.sort_values(by='Date')

    df = df.fillna(df.mean())

    df['month'] = df['Date'].dt.month
    df['day'] = df['Date'].dt.day
    df['hour'] = df['Date'].dt.hour
    df['day_of_week'] = df['Date'].dt.dayofweek

    df = df[['Date', 'month', 'day', 'hour', 'baropressure', \
             'humidity', 'temperature', 'winddirection', \
             'windspeed', 'n', 'day_of_week', 'ST']]
    return df

