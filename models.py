import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import LearningCurveDisplay, ShuffleSplit
from sklearn.linear_model import SGDRegressor, LogisticRegression, SGDClassifier, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_digits
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error

from preprocessing import read_daily_data

COLUMNS, FEATURES, X, y = read_daily_data()
std = StandardScaler()
X_std = std.fit_transform(X)
# sgd = SGDRegressor(loss="epsilon insensitive")
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
plt.show()