import numpy as np
import pandas as pd

NAME = {
    "Temperature": "T",
    "Precipitation Total": "PT",
    "Relative Humidity": "RH",
    "Wind Speed": "WS",
    "Wind Direction": "WD",
    "Cloud Cover Total": "CCT",
    "Mean Sea Level Pressure": "MSLP"
}

def get_data():
    '''
        Return the data without headers and with date index.
    '''
    data = pd.read_csv("data/dataexport_20231120T214607.csv", dtype="string")
    data.columns = data.iloc[3].rename("Variable")
    data = data.iloc[9:]
    _timestamp = data.iloc[:, 0].astype("datetime64[s]")
    _yaer = _timestamp.dt.year.rename("Year")
    _month = _timestamp.dt.month.rename("Month")
    _day = _timestamp.dt.day.rename("Day")
    _hour = _timestamp.dt.hour.rename("Hour")
    data.set_index([_yaer, _month, _day, _hour], inplace=True)
    data = data.iloc[:, 1:]
    return data.astype("float64").rename(columns=NAME)

def drop_missing_data(data):
    '''
        Remove last columns and return the number of them.
    '''
    missind_data = data.shape[0]
    data.dropna(inplace=True)
    missind_data -= data.shape[0]
    data.drop(data.tail(data.shape[0] % 24).index,inplace=True)
    return missind_data

def change_resolution_to_daily(data):
    '''
        Calculate mean of values in each day.
    '''
    daily_data = pd.DataFrame([[np.mean(data.iloc[i - 23:i + 1, j])
        for j in range(data.shape[1])] for i in range(23, data.shape[0], 24)],
        columns=data.columns)
    daily_data.rename(columns={"T": "MEANT"}, inplace=True)
    daily_data["MAXT"] = pd.Series([np.max(data.iloc[i - 23:i + 1, 0])
        for i in range(23, data.shape[0], 24)])
    daily_data["MINT"] = pd.Series([np.min(data.iloc[i - 23:i + 1, 0])
        for i in range(23, data.shape[0], 24)])
    daily_data["PT"] = pd.Series([np.sum(data.iloc[i - 23:i + 1, 1])
        for i in range(23, data.shape[0], 24)])
    _timestamp = data.index[::24]
    _yaer = pd.Series([u[0] for u in _timestamp], name="Year")
    _month = pd.Series([u[1] for u in _timestamp], name="Month")
    _day = pd.Series([u[2] for u in _timestamp], name="Day")
    daily_data.set_index([_yaer, _month, _day], inplace=True)
    return daily_data

def write_daily_data(daily_data):
    '''
        For saving time, write daily data in "data/daily_data.csv".
    '''
    daily_data.to_csv("data/daily_data.csv")

def read_daily_data():
    '''
        Read file "data/daily_data.csv". Return COLUMNS, FEATURES, X, y.
    '''
    df = pd.read_csv("data/daily_data.csv", index_col=["Year", "Month", "Day"], header=0)
    FEATURES = df.drop(columns=["PT"]).columns
    COLUMNS = df.columns
    y = df.pop("PT")
    X = df
    return COLUMNS, FEATURES, X, y

COLUMNS, FEATURES, _, _ = read_daily_data()