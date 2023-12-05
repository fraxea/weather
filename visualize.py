import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from getdata import read_daily_data, FEATURES

def show_histogram(X, titles=FEATURES, row=2, column=4):
    fig, ax = plt.subplots(row, column, subplot_kw=dict(box_aspect=1))
    index_title = 0
    for i,ax_row in enumerate(ax):
        for j,axes in enumerate(ax_row):
            axes.set_title(titles[index_title])
            axes.hist(X[:, index_title], bins=365)
            index_title += 1
    fig.tight_layout()
    fig.set_size_inches(fig.get_size_inches() * 2)

# _, _, X, y = read_daily_data()
# X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.3, random_state=42)
# stnd_s = StandardScaler()
# X_train_scaled = stnd_s.fit_transform(X_train)
# pca = PCA().fit(X_train)
# scaled_pca = PCA().fit(X_train_scaled)
# X_train_transformed = pd.DataFrame(pca.transform(X_train), index=X_train.index, columns=FEATURES)
# X_train_std_transformed = pd.DataFrame(scaled_pca.transform(X_train_scaled), index=X_train.index, columns=FEATURES)
# first_pca_component = pd.DataFrame(
#     pca.components_[0], index=FEATURES, columns=["Without Scaling"]
# )
# first_pca_component["With Scaling"] = scaled_pca.components_[0]
# first_pca_component.plot.bar(
#     title="Weights of the first principal component"
# )
# print(scaled_pca.explained_variance_ratio_)

# print(scaled_pca.singular_values_)
# print(scaled_pca.explained_variance_ratio_)
# plt.show()