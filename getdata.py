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

COLUMNS, FEATURES, X, y = read_daily_data()