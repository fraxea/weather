import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from preprocessing import read_daily_data

def show_histogram(data):
    for i in range(2, data.shape[1]):
        plt.subplot(2, 2, i)
        plt.hist(data.iloc[:, i], 365)
        plt.title(data.columns[i], pad=1)
    plt.subplot(2, 2, 1)
    plt.hist(data.iloc[:, 0], 365)
    plt.title(data.columns[0], pad=1)

x = np.random.normal(2.5, 1, 1000)
y = np.random.uniform(2.5, 2, 1000)
plt.scatter(x, y)
# daily_daat = read_daily_data()
# show_histogram(daily_daat)
plt.show()