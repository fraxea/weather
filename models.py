import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score

from getdata import read_daily_data
from preprocessing import split_data

X, y = read_daily_data()
X_train, X_test, y_train, y_test = split_data(X, y)
svr = SVR()
model = make_pipeline(
    StandardScaler(),
    # rfr
    svr
)
model.fit(X_train, y_train)
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)
print(f"Train score:    {r2_score(y_pred=y_pred_train, y_true=y_train)}")
print(f"Test score:     {r2_score(y_pred=y_pred_test, y_true=y_test)}")