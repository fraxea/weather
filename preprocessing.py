import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

# print(y[y > 0.1].shape[0] / y.shape[0])
# print(y.sum() / (y.shape[0] / 365.25))

# hist, bin_edges = np.histogram(y, bins=[0, 0.1, y.max() + 0.1])
# pd_hist = pd.Series(hist)
# fig = pd_hist.plot(kind="bar", ylabel="number of days", width=0.5)
# fig.set_xticklabels(["not rainy", "rainy"], rotation="horizontal")
# plt.show()