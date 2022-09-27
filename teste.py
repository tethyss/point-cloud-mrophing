import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

full_data = pd.read_csv('./data.csv', header=0)
full_data = full_data.values
full_data[:, 2:27] = StandardScaler().fit_transform(full_data[:, 2:27])
deposits = pd.read_csv('./deposit.csv', header=0)
deposits = deposits.values
deposits = full_data[(335-deposits[:, 1].astype(int)) * 335 + deposits[:, 0].astype(int)-1]
# generate random point
nongranite = np.argwhere(full_data[:, 29] == 0)
nongranite = np.reshape(full_data[nongranite], [len(nongranite), 30])
rand = np.random.randint(len(nongranite), size=49)
nongranite = nongranite[rand]
rand = np.random.randint(len(full_data), size=120)
random_point = full_data[rand]
landmark_points = np.vstack((deposits, nongranite, random_point))
landmark_points = np.unique(landmark_points, axis=0)

pass
