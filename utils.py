import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def load_data():
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
    i = list(data.columns)
    i[0], i[1] = i[1], i[0]
    data = data[i].astype("float64").rename(columns={
        "Temperature": "T",
        "Precipitation Total": "PT",
        "Relative Humidity": "RH",
        "Wind Speed": "WS",
        "Wind Direction": "WD",
        "Cloud Cover Total": "CCT",
        "Mean Sea Level Pressure": "MSL"
    })
    return data

def write_daily_data(daily_data):
    '''
        For saving time, write daily data in "data/daily_data.csv".
    '''
    daily_data.to_csv("data/daily_data.csv")

def read_daily_data():
    '''
        Read file "data/daily_data.csv". Return X, y.
    '''
    df = pd.read_csv("data/daily_data.csv", index_col=["Year", "Month", "Day"], header=0)
    y = df.pop("PT")
    X = df
    return X, y


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


def show_histogram(X, titles, row=2, column=3):
    fig, ax = plt.subplots(row, column, subplot_kw=dict(box_aspect=1))
    index_title = 0
    for i,ax_row in enumerate(ax):
        for j,axes in enumerate(ax_row):
            axes.set_title(titles[index_title])
            axes.hist(X[:, index_title], bins=365)
            index_title += 1
    fig.tight_layout()
    fig.set_size_inches(fig.get_size_inches() * 2)
    fig.suptitle("Histogram", fontsize=16)


def search_results(search):
    results_df = pd.DataFrame(search.cv_results_)
    results_df = results_df.sort_values(by=["rank_test_score"])
    results_df = results_df.set_index(
        results_df["params"].apply(lambda x: "_".join(str(val) for val in x.values()))
    ).rename_axis("hyper params")
    return results_df[["rank_test_score", "mean_test_score"]]

def score_model(score, y_train, y_train_pred, y_test, y_test_pred, name):
    return pd.DataFrame(
        [[score(y_train, y_train_pred), score(y_test, y_test_pred)]],
        columns=["Train Score", "Test Score"], index=[name]
    )
