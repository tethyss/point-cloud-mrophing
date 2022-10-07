import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


def read_data(plot=1):
    full_data = pd.read_csv('./data.csv', header = 0)
    full_data = full_data.values
    full_data[:, 2:27] = StandardScaler().fit_transform(full_data[:, 2:27])
    deposits = pd.read_csv('./deposit.csv', header = 0)
    deposits = deposits.values
    deposits = full_data[(335 - deposits[:, 1].astype(int)) * 335 + deposits[:, 0].astype(int) - 1]
    # generate random point
    nongranite = np.argwhere(full_data[:, 29] == 0)
    nongranite = np.reshape(full_data[nongranite], [len(nongranite), 30])
    rand = np.random.randint(len(nongranite), size = 49)
    nongranite = nongranite[rand]
    rand = np.random.randint(len(full_data), size = 122)
    random_point = full_data[rand]
    if plot == 1:
        plt.figure(figsize = (15, 15))
        data_show = np.reshape(full_data[:, 29], [335, 335])
        plt.imshow(data_show[::-1], origin = 'lower', cmap = 'Pastel2')
        plt.scatter(random_point[:, 0], random_point[:, 1], c = 'b', s = 200, marker = '1', label = "random points")
        plt.scatter(deposits[:, 0], deposits[:, 1], c = 'g', s = 200, marker = '^', label = "deposits")
        plt.scatter(nongranite[:, 0], nongranite[:, 1], s = 200, marker = 'x', label = "non-deposits")
        plt.legend(fontsize = 20)
        plt.show()
    landmark_points = np.vstack((deposits, nongranite, random_point))
    landmark_points = np.unique(landmark_points, axis = 0)
    order = np.arange(landmark_points.shape[0])
    np.random.shuffle(order)
    landmark_points = landmark_points[order, :]
    return full_data[:, :27], landmark_points[:200, :27]


def a2g(rawdata, file):
    columns = ['X', 'Y', 'Ag', 'Al', 'Au', 'B', 'Ba', 'Be', 'Bi', 'Ca', 'Co', 'F', 'Fe', 'K', 'La', 'Li', 'Mg',
               'Mn', 'Mo', 'Nb', 'P', 'Sn', 'Sr', 'Ti', 'V', 'Y1', 'Zr']
    title = np.asarray('data')
    col = np.asarray(columns).reshape((-1, 1))
    head = np.vstack((title, col))
    head = pd.DataFrame(head)
    rawdata = pd.DataFrame(rawdata)
    head.to_csv(file, index=False, header=False)
    rawdata.to_csv(file, index=False, header=False, mode="a")


data, landmarks = read_data(plot=0)
a2g(data, '1test.dat')
pass
