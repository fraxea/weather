import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from getdata import read_daily_data
from preprocessing import split_data

# X, y = read_daily_data()
# X_train, X_test, y_train, y_test = split_data(X, y)
