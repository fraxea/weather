import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def get_data():
    raw_data = pd.read_csv("data/dataexport_20231120T214607.csv", dtype="string")
    no_headers = raw_data[9:]
    col_dict = dict(zip(raw_data.columns, raw_data.iloc[8]))
    ix_dict = dict(zip(raw_data.index, raw_data.iloc[:, 0]))
    renamed_data = no_headers.rename(columns=col_dict, index=ix_dict)
    cleaned_data = renamed_data.iloc[:, 1:].dropna().astype(np.float32)
    missing_data = renamed_data.shape[0] - cleaned_data.shape[0]
    return missing_data, cleaned_data

def get_daily_temperature(cleaned_data):
    temperature = cleaned_data.iloc[:, 0]
    MIN_T = temperature.min()
    MAX_T = temperature.max()
    MEAN_T = temperature.mean()
    n_examples = temperature.shape[0]
    mean_temperature = np.array([np.average(temperature[i:i + 24]) for i in range(0, n_examples, 24)])
    min_temperature = np.array([np.min(temperature[i:i + 24]) for i in range(0, n_examples, 24)])
    max_temperature = np.array([np.max(temperature[i:i + 24]) for i in range(0, n_examples, 24)])
    var_temperature = np.array([np.var(temperature[i:i + 24]) for i in range(0, n_examples, 24)])
    return MIN_T, MAX_T, MEAN_T, mean_temperature, min_temperature, max_temperature, var_temperature

def get_daily_date(cleaned_data):
    days = pd.date_range(cleaned_data.index[0], cleaned_data.index[-1])
    return days.date

_, cleaned_data = get_data()
cleaned_data.drop(cleaned_data.columns[4], inplace=True, axis=1)
print(get_daily_date(cleaned_data))

# def get_train_test(cleaned_data):
#     y = cleaned_data.pop(cleaned_data.columns[1])
#     X_train, X_test, y_train, y_test = train_test_split(
#         cleaned_data, y, test_size=0.30, random_state=42)
#     return X_train, X_test, y_train, y_test

# X_train, X_test, y_train, y_test = get_train_test(cleaned_data)
# print(X_train.shape)
# print(X_test.shape)
# print(y_train.shape)
# print(y_test.shape)
