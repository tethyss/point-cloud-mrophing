import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import probscale
from sklearn.preprocessing import StandardScaler


def read_data(plot=1):
    full_data = pd.read_csv('./data.csv', header = 0)
    full_data = full_data.values
    full_data[:, 2:27] = StandardScaler().fit_transform(full_data[:, 2:27])
    return full_data[:, :27]


data = read_data()
result_container = np.load("result.npy")
result_cdf_container = np.load('result_cdf.npy')
print('result_mean = ', np.mean(result_container[:, 10, :]))
print('data_mean = ', np.mean(data[:, 12]))
print('result_std = ', np.std(result_container[:, 10, :]))
print('data_std = ', np.std(data[:, 12]))

pass