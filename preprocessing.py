import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

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
    return data.astype("float64")

# def get_pressure_reversed_temperature(data):
    temperature = data["Temperature"]
    pressure = data["Mean Sea Level Pressure"]
    temperature.dropna(inplace=True)
    pressure.dropna(inplace=True)
    return temperature, pressure

# def get_data():
    raw_data = pd.read_csv("data/dataexport_20231120T214607.csv", dtype="string")
    no_headers = raw_data[9:]
    col_dict = dict(zip(raw_data.columns, raw_data.iloc[8]))
    ix_dict = dict(zip(raw_data.index, raw_data.iloc[:, 0]))
    renamed_data = no_headers.rename(columns=col_dict, index=ix_dict)
    cleaned_data = renamed_data.iloc[:, 1:].dropna().astype(np.float32)
    missing_data = renamed_data.shape[0] - cleaned_data.shape[0]
    return missing_data, cleaned_data

def drop_missing_data(data):
    '''
        Remove last columns and return the number of them.
    '''
    missind_data = data.shape[0]
    data.dropna(inplace=True)
    missind_data -= data.shape[0]
    return missind_data

def remove_irrelevant_columns(data):
    '''
        Remove Wind Direction and Pressure.
    '''
    data.drop(columns=["Wind Direction", "Mean Sea Level Pressure"], inplace=True)
    return data

def show_duplicates(data):
    '''
        Shows how many duplicate rows and how many duplicate value for each parameter we have.
    '''
    total = data.duplicated().value_counts()
    single = pd.DataFrame(dict(zip(data.columns, [data.iloc[:, i].duplicated().value_counts() for i in range(data.shape[1])])))
    return total, single

def change_resolution_to_daily(data):
    '''
        Calculate mean of values in each day.
    '''
    daily_data_meaned = pd.DataFrame([[np.mean(data.loc[data.index[i][:3], data.columns[j]])
        for j in range(data.shape[1])] for i in range(0, data.shape[0], 24)],
        columns=data.columns)
    _timestamp = data.index[::24]
    _yaer = pd.Series([u[0] for u in _timestamp], name="Year")
    _month = pd.Series([u[1] for u in _timestamp], name="Month")
    _day = pd.Series([u[2] for u in _timestamp], name="Day")
    daily_data_meaned.set_index([_yaer, _month, _day], inplace=True)
    return daily_data_meaned

def split_data(normalized_data):
    '''
        Split data to first approximately 7 years test and remaining train.
    '''
    y = normalized_data.pop("Precipitation Total")
    X = normalized_data
    return train_test_split(X, y, test_size=0.3, shuffle=False)

data = get_data()
drop_missing_data(data)
remove_irrelevant_columns(data)
show_duplicates(data)
daily_data = change_resolution_to_daily(data)
# X_train, X_test, y_train, y_test = split_data(normalized_data)
# print(X_train)
# print(X_test)
# print(y_train)
# print(y_test)