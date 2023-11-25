import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from preprocessing import get_daily_temperature, get_data, get_daily_precipitation

def plot_temperature(temperature, c):
    compressed = pd.Series([np.average(temperature[i:-1:365]) for i in range(365)], index=pd.date_range("20140101", "20141231", freq='D'))
    compressed.plot()
    plt.plot(compressed.index, compressed, c)

_, data = get_data()
MIN_T, MAX_T, MEAN_T, mean_temperature, min_temperature, max_temperature, var_temperature = get_daily_temperature(data)
plot_temperature(mean_temperature, 'b')
plot_temperature(min_temperature, 'g')
plot_temperature(max_temperature, 'r')
mean_precipitation, min_precipitation, max_precipitation, var_precipitation = get_daily_precipitation(data)
plot_temperature(max_precipitation, 'm')
plt.show()