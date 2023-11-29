import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from preprocessing import read_daily_data, split_data, COLUMNS

def show_histogram(data, titles=COLUMNS, row=3, column=3):
    fig, ax = plt.subplots(row, column)
    # fig, ax = plt.subplots(row, column, subplot_kw=dict(box_aspect=1))
    for i,ax_row in enumerate(ax):
        for j,axes in enumerate(ax_row):
            axes.set_title(titles[i * column + j])
            y = data[:, i * row + j]
            axes.hist(y, bins=365)
    fig.tight_layout()

_, _, data = read_daily_data()
# X_train, X_test, y_train, y_test = split_data(data)
# print(type(X_train))
show_histogram(data.values)
plt.show()