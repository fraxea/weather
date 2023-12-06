import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import LearningCurveDisplay, learning_curve, train_test_split
from sklearn.linear_model import SGDRegressor, LogisticRegression
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from keras.models import Sequential
# from keras.activations import SG
from keras.layers import Dense, Input

from getdata import read_daily_data

def naive_bayes(X, y):
    '''
        Given X and y, return test accuracy, train accuracy, and sample size table.
    '''
    svr = GradientBoostingRegressor(random_state=42)
    train_sizes, train_scores, test_scores = learning_curve(svr, X, y, cv=5)
    plt.close()
    display = LearningCurveDisplay(train_sizes=train_sizes,
        train_scores=train_scores, test_scores=test_scores, score_name="Score")
    display.plot()
    accuracy_table = pd.DataFrame(np.array([[train_size for train_size in train_sizes]
        , [cv_train_scores.mean() for cv_train_scores in train_scores]
        , [cv_test_scores.mean() for cv_test_scores in test_scores]]).swapaxes(0, 1)
        , columns=["Train Size", "Train Score", "Test Score"])
    return accuracy_table

def svm_model(X, y):
    '''
        Given X and y, return test accuracy, train accuracy, and sample size table.
    '''
    svr = SVR(kernel="rbf")
    train_sizes, train_scores, test_scores = learning_curve(svr, X, y, cv=5)
    plt.close()
    display = LearningCurveDisplay(train_sizes=train_sizes,
        train_scores=train_scores, test_scores=test_scores, score_name="Score")
    display.plot()
    accuracy_table = pd.DataFrame(np.array([[train_size for train_size in train_sizes]
        , [cv_train_scores.mean() for cv_train_scores in train_scores]
        , [cv_test_scores.mean() for cv_test_scores in test_scores]]).swapaxes(0, 1)
        , columns=["Train Size", "Train Score", "Test Score"])
    return accuracy_table

def logistic_model(X, y):
    '''
        Given X and y, return test accuracy, train accuracy, and sample size table.
    '''
    lgs_reg = SGDRegressor("epsilon_insensitive", random_state=42)
    train_sizes, train_scores, test_scores = learning_curve(lgs_reg, X, y, cv=5)
    plt.close()
    display = LearningCurveDisplay(train_sizes=train_sizes,
        train_scores=train_scores, test_scores=test_scores, score_name="Score")
    display.plot()
    accuracy_table = pd.DataFrame(np.array([[train_size for train_size in train_sizes]
        , [cv_train_scores.mean() for cv_train_scores in train_scores]
        , [cv_test_scores.mean() for cv_test_scores in test_scores]]).swapaxes(0, 1)
        , columns=["Train Size", "Train Score", "Test Score"])
    return accuracy_table

def linear_model(X, y):
    '''
        Given X and y, return test accuracy, train accuracy, and sample size table.
    '''
    sgd_reg = SGDRegressor(random_state=42)
    train_sizes, train_scores, test_scores = learning_curve(sgd_reg, X, y, cv=5)
    plt.close()
    display = LearningCurveDisplay(train_sizes=train_sizes,
        train_scores=train_scores, test_scores=test_scores, score_name="Score")
    display.plot()
    accuracy_table = pd.DataFrame(np.array([[train_size for train_size in train_sizes]
        , [cv_train_scores.mean() for cv_train_scores in train_scores]
        , [cv_test_scores.mean() for cv_test_scores in test_scores]]).swapaxes(0, 1)
        , columns=["Train Size", "Train Score", "Test Score"])
    return accuracy_table
# # COLUMNS, FEATURES,
# _, _, X, y = read_daily_data()
# X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.3, random_state=42)
# model = Sequential([
#     Dense(4, activation="relu"),
#     Dense(3, activation="relu"),
#     Dense(2, activation="relu"),
#     Dense(1, activation="relu")
# ])
# model.compile(loss="mean_squared_error")
# std = StandardScaler()
# std.fit(X)
# X_train_std = std.transform(X_train)
# model.fit(X_train_std, y_train, epochs=100)
# X_test_std = std.transform(X_test)
# y_pred = model.predict(X_test_std)
# score = r2_score(y_test, y_pred)
# print(score)
# std = StandardScaler()
# std.fit(X)
# X_train_std = std.transform(X_train)
# sgd_reg = SGDRegressor(random_state=42)
# sgd_reg.fit(X_train_std, y_train)
# X_test_std = std.transform(X_test)
# sgd_score = sgd_reg.score(X_test_std, y_test)
# print(f"{sgd_score:.3f}")
# lgs_reg = LogisticRegression(random_state=42)
# lgs_reg.fit(X_train_std, y_train)
# lgs_score = lgs_reg.score(X_test_std, y_test)
# lgs_reg = LogisticRegression(random_state=42)
# lgs_reg.fit(X_train, y_train)
# y_predict = lgs_reg.predict(X_test)
# print(y_predict)
# X_std = StandardScaler().fit_transform(X)
# accuracy_table = naive_bayes(X_std, y)
# print(accuracy_table)
# plt.show()
'''# sgd = SGDRegressor(loss="epsilon insensitive")
# svc = SVC(kernel="rbf", gamma=0.001)
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 6), sharey=True)

sgd_reg = SGDRegressor(random_state=42)
sgd_reg2 = LinearRegression()
# lgs_reg = LogisticRegression(max_iter=100, random_state=42)
# sgd_cls = SGDClassifier(loss="log_loss", alpha=0, max_iter=1000, random_state=42)

common_params = {
    "X": X_std,
    "y": y,
    "train_sizes": np.linspace(0.1, 1.0, 5),
    "cv": ShuffleSplit(n_splits=50, test_size=0.2, random_state=0),
    "score_type": "both",
    "n_jobs": 4,
    "line_kw": {"marker": "o"},
    "std_display_style": "fill_between",
    "score_name": "Accuracy",
}

for ax_idx, estimator in enumerate([sgd_reg, sgd_reg2]):
    LearningCurveDisplay.from_estimator(estimator, **common_params, ax=ax[ax_idx])
    handles, label = ax[ax_idx].get_legend_handles_labels()
    ax[ax_idx].legend(handles[:2], ["Training Score", "Test Score"])
    ax[ax_idx].set_title(f"Learning Curve for {estimator.__class__.__name__}")
plt.show()'''