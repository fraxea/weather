import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def get_data():
    raw_data = pd.read_csv("data/dataexport_20231120T214607.csv", dtype="string")
    no_headers = raw_data[9:]
    col_dict = dict(zip(raw_data.columns, raw_data.iloc[8]))
    ix_dict = dict(zip(raw_data.index, raw_data.iloc[:, 0]))
    renamed_data = no_headers.rename(columns=col_dict, index=ix_dict)
    cleaned_data = renamed_data.iloc[:, 1:].dropna().astype(np.float32)
    missing_data = renamed_data.shape[0] - cleaned_data.shape[0]
    return missing_data, cleaned_data

_, q = get_data()
print(q.columns)
print(q.shape)