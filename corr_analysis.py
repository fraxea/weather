"""
This script is used to perform correlation analysis on the data.
"""
import os
import matplotlib.pyplot as plt
import seaborn as sns

from utils import load_data, drop_missing_data, change_resolution_to_daily

data = load_data()
drop_missing_data(data)
daily_data = change_resolution_to_daily(data)
corr_matrix = daily_data.corr()
# save figure in "figures/correlation_matrix.png"
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm")
plt.title("Correlation Matrix")
# Create the directory if it does not exist
if not os.path.exists("figures"):
    os.makedirs("figures")
plt.savefig("figures/correlation_matrix.png")
