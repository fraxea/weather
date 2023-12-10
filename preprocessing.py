import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def normalization(X_train, X_test):
    '''
        Standardize features by removing the mean and scaling to unit variance.
        Return X_train_std, X_test_std
    '''
    std = StandardScaler().fit(X_train)
    X_train_std = std.transform(X_train)
    X_test_std = std.transform(X_test)
    return X_train_std, X_test_std

def split_data(X, y):
    '''
        Split X and y into random train and test subsets.
        Return X_train, X_test, y_train, y_test
    '''
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, shuffle=True)
    return X_train, X_test, y_train, y_test

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
        Calculate max, min, and mean of parameters in each day.
        For Precipitation sum is calculated.
    '''
    mean_data = pd.DataFrame([[np.mean(data.iloc[i - 23:i + 1, j])
        for j in range(1, data.shape[1])] for i in range(23, data.shape[0], 24)],
        columns=[name + "MEAN" for name in data.columns[1:]])
    max_data = pd.DataFrame([[np.max(data.iloc[i - 23:i + 1, j])
        for j in range(1, data.shape[1])] for i in range(23, data.shape[0], 24)],
        columns=[name + "MAX" for name in data.columns[1:]])
    min_data = pd.DataFrame([[np.min(data.iloc[i - 23:i + 1, j])
        for j in range(1, data.shape[1])] for i in range(23, data.shape[0], 24)],
        columns=[name + "MIN" for name in data.columns[1:]])
    daily_data = pd.DataFrame([[np.sum(data.iloc[i - 23:i + 1, 0])]
        for i in range(23, data.shape[0], 24)], columns=data.columns[0:1])
    daily_data = pd.concat([daily_data, mean_data, max_data, min_data], axis=1)
    # set Index
    _timestamp = data.index[::24]
    _yaer = pd.Series([u[0] for u in _timestamp], name="Year")
    _month = pd.Series([u[1] for u in _timestamp], name="Month")
    _day = pd.Series([u[2] for u in _timestamp], name="Day")
    daily_data.set_index([_yaer, _month, _day], inplace=True)
    return daily_data

# hist, bin_edges = np.histogram(y, bins=[0, 0.1, y.max() + 0.1])
# pd_hist = pd.Series(hist)
# fig = pd_hist.plot(kind="bar", ylabel="number of days", width=0.5)
# fig.set_xticklabels(["not rainy", "rainy"], rotation="horizontal")
# plt.show()