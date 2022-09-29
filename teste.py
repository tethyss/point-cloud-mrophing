import numpy as np
import ot
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


def read_data(plot=1):
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
    if plot == 1:
        plt.figure(figsize=(15, 15))
        data_show = np.reshape(full_data[:, 29], [335, 335])
        plt.imshow(data_show[::-1], origin='lower', cmap='Pastel2')
        plt.scatter(random_point[:, 0], random_point[:, 1], c='b', s=200, marker='1', label="random points")
        plt.scatter(deposits[:, 0], deposits[:, 1], c='g', s=200, marker='^', label="deposits")
        plt.scatter(nongranite[:, 0], nongranite[:, 1], s=200, marker='x', label="non-deposits")
        plt.legend(fontsize=20)
        plt.show()
    landmark_points = np.vstack((deposits, nongranite, random_point))
    landmark_points = np.unique(landmark_points, axis=0)
    return full_data[:, :27], landmark_points[:200, :27]


def convert_to_cdf(data1, if_show=0, show_config=None, color='b'):  # '#F9E855'
    if show_config is None:
        show_config = [10, 15]
    if if_show == 1:
        plt.figure(1)
        plt.title('raw data')
        plt.scatter(data1[:, show_config[0]], data1[:, show_config[1]], s=10, c=color)  # '#FF1F5B'
        plt.axis('square')
    p = 1. * np.arange(len(data1)+2) / (len(data1) + 1)
    for ele in range(len(data1[1])):
        data_sorted = data1[:, ele]
        data_sorted = np.hstack(
            (data_sorted.reshape([len(data1), 1]), np.arange(len(data1)).reshape([len(data1), 1])))
        idex = np.argsort(data_sorted, axis=0)
        data_sorted = data_sorted[idex[:, 0]]
        data_sorted[:, [0, 1]] = data_sorted[:, [1, 0]]
        data_sorted = np.hstack((data_sorted, p[1:len(data1)+1].reshape([len(data1), 1])))
        idex = np.argsort(data_sorted, axis=0)
        data_sorted = data_sorted[idex[:, 0]]
        data1[:, ele] = data_sorted[:, 2]
    if if_show == 1:
        plt.figure(2)
        plt.title('CDF')
        plt.scatter(data1[:, show_config[0]], data1[:, show_config[1]], s=10, c=color)
        plt.axis('square')
        plt.show()
    return data1


def transport(lm_cdf, if_show=0, show_config=None):
    x = np.random.normal(0, 1, len(lm_cdf[:, 1])).reshape((len(lm_cdf[:, 1]), 1))
    for e in range(len(lm_cdf[1, :]) - 1):
        x = np.hstack((x, np.random.normal(0, 1, len(lm_cdf[:, 1])).reshape((len(lm_cdf[:, 1]), 1))))
    a, b = np.ones((len(lm_cdf),)) / len(lm_cdf), np.ones((len(lm_cdf),)) / len(lm_cdf)
    x_cdf = convert_to_cdf(np.copy(x), if_show=if_show, show_config=show_config, color='r')
    dist_matrix = ot.dist(lm_cdf, x_cdf)
    pair = ot.emd(a, b, dist_matrix)
    x_cdf = x_cdf[np.nonzero(pair)[1]]
    x = x[np.nonzero(pair)[1]]
    if if_show == 1:
        plt.plot([lm_cdf[:, show_config[0]], x_cdf[:, show_config[0]]],
                 [lm_cdf[:, show_config[1]], x_cdf[:, show_config[1]]], c=[.5, .5, 1], alpha=0.2)
        plt.plot(lm_cdf[:, show_config[0]], lm_cdf[:, show_config[1]], '+', c='b', label='Source samples')
        plt.plot(x_cdf[:, show_config[0]], x_cdf[:, show_config[1]], 'x', c='r', label='Target samples')
        plt.legend(loc=0)
        plt.title('OT matrix with samples')
        plt.axis('square')
        plt.show()
    return x, x_cdf


data, landmarks = read_data(plot=1)
landmarks_cdf = convert_to_cdf(np.copy(landmarks[:, 2:]), show_config=[10, 15], if_show=1)
mf_raw, mf_cdf = transport(np.copy(landmarks_cdf), if_show=1, show_config=[10, 15])

pass
