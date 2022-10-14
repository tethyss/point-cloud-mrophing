from __future__ import annotations
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist  # type: ignore
import random
import ot
import os
import math


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


def variogram_gam(data, grid, cellsize, nlag):
    a2g(data, "gam.dat")

    with open("gam.par", "w") as f:
        f.write("                         Parameters for GAM                                  \n")
        f.write("                         *******************                                 \n")
        f.write("                                                                             \n")
        f.write("START OF PARAMETERS:                                                         \n")
        f.write("gam.dat                                 -file with data                      \n")
        f.write("25 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 -number of var.,col numbers\n")
        f.write("-1.0e21     1.0e21                      -trimming limits                     \n")
        f.write("gam_out.out                             -file for variogram output           \n")
        f.write("1                                       -grid or realization number          \n")
        f.write(str(grid[0]) + " 0.5 " + str(cellsize) + " -nx, xmn, xsiz                       \n")
        f.write(str(grid[1]) + " 0.5 " + str(cellsize) + " -ny, ymn, ysiz                       \n")
        f.write("1 0 0                                   -nz, zmn, zsiz                       \n")
        f.write("1 " + str(nlag) + "                     -number of directions, number of lags\n")
        f.write("1  0  0                                 -ixd(1),iyd(1),izd(1)                \n")
        f.write("0                                       -standardize sill? (0=no, 1=yes)     \n")
        f.write("325                                     -number of variograms                \n")
        for v1 in range(1, 26):
            for v2 in range(v1, 26):
                f.write(str(v1) + " " + str(v2) + " 2      -tail, head, variogram type  \n")
    os.system("gam.exe gam.par")

    lag = np.arange(1, int(nlag + 1), dtype = float).reshape((nlag, 1))
    i = -1
    gamma = np.empty((nlag, 325))
    with open("gam_out.out") as f:
        for line in f:
            if line[0] == "C":
                i += 1
                n = 0
            else:
                _, _, g, *_ = line.split()
                gamma[n, i] = float(g)
                n += 1
    gamma = np.hstack((lag, lag, gamma))

    return gamma


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


def convert_to_cdf(data1, if_show=0, show_config=None, color='b'):  # '#F9E855'
    # if show_config is None:
    #     show_config = [10, 15]
    # if if_show == 1:
    #     plt.figure(1)
    #     plt.title('raw data')
    #     plt.scatter(data1[:, show_config[0]], data1[:, show_config[1]], s = 10, c = color)  # '#FF1F5B'
    #     plt.axis('square')
    p = 1. * np.arange(len(data1) + 2) / (len(data1) + 1)
    for ele in range(len(data1[1])):
        data_sorted = data1[:, ele]
        data_sorted = np.hstack(
            (data_sorted.reshape([len(data1), 1]), np.arange(len(data1)).reshape([len(data1), 1])))
        idex = np.argsort(data_sorted, axis = 0)
        data_sorted = data_sorted[idex[:, 0]]
        data_sorted[:, [0, 1]] = data_sorted[:, [1, 0]]
        data_sorted = np.hstack((data_sorted, p[1:len(data1) + 1].reshape([len(data1), 1])))
        idex = np.argsort(data_sorted, axis = 0)
        data_sorted = data_sorted[idex[:, 0]]
        data1[:, ele] = data_sorted[:, 2]
    # if if_show == 1:
    #     plt.figure(2)
    #     plt.title('CDF')
    #     plt.scatter(data1[:, show_config[0]], data1[:, show_config[1]], s = 10, c = color)
    #     plt.axis('square')
    #     plt.show()
    return data1


def variogram_calculation(data, dist_matrix, lag, steps, tol, channels):
    # variogram structure: 0: Average distance; 1: counts in lag.
    variogram = np.zeros((steps + 1, int(2 + channels * (channels + 1) / 2)))
    for i_Step in range(steps):
        print("i_Step :", i_Step + 1)
        x_pos, y_pos = np.where((dist_matrix > lag * (i_Step + 1) - tol) & (dist_matrix < lag * (i_Step + 1) + tol))
        for j in range(len(x_pos)):
            x_j, y_j = x_pos[j], y_pos[j]
            variogram[i_Step + 1, 1] += 1
            variogram[i_Step + 1, 0] += dist_matrix[x_j, y_j]
            dif_j = data[x_j, 2:] - data[y_j, 2:]
            pos = 2
            for c in range(channels):
                variogram[i_Step + 1, int(pos):int(pos + len(dif_j[c:]))] += dif_j[c] * dif_j[c:]
                pos += len(dif_j[c:])
        variogram[i_Step + 1, 2:] = variogram[i_Step + 1, 2:] / (2 * variogram[i_Step + 1, 1])
        variogram[i_Step + 1, 0] = variogram[i_Step + 1, 0] / variogram[i_Step + 1, 1]  # Average distance
        variogram[i_Step + 1, 1] = variogram[i_Step + 1, 1] / 2  # Without duplicates
    return variogram


def plot_variogram(variogram, color="green"):
    names = {'1': 'Ag', '2': 'Al', '3': 'Au', '4': 'B',
             '5': 'Ba', '6': 'Be', '7': 'Bi', '8': 'Ca',
             '9': 'Co', '10': 'F', '11': 'Fe', '12': 'K',
             '13': 'La', '14': 'Li', '15': 'Mg', '16': 'Mn',
             '17': 'Mo', '18': 'Nb', '19': 'P', '20': 'Sn',
             '21': 'Sr', '22': 'Ti', '23': 'V', '24': 'Y',
             '25': 'Zr'}
    fig, axs = plt.subplots(5, 5, figsize = (17, 14))
    plt.suptitle('Direct variogram', size = 20)
    plt.subplots_adjust(left = 0.05, bottom = 0.05, right = 0.95, top = 0.95, wspace = 0.5, hspace = 0.3)
    for col in range(5):
        for row in range(5):
            axs[row, col].set_ylim(0.0, 1.5)
    if len(variogram.shape) == 2:
        for i in range(25):
            axs[int(i % 5), int(i / 5)].plot(variogram[:, 0], variogram[:, int((51 - i) * i / 2 + 2)], linestyle = '--',
                                             marker = 'x', markersize = 0.5, linewidth = 0.8,
                                             color = color, label = 'Samples', )
            axs[int(i % 5), int(i / 5)].set_xlabel('Distance')
            axs[int(i % 5), int(i / 5)].set_ylabel("%s" % (names[str(i + 1)]), labelpad = 0, size = 20)
            # axs[int(i % 5), int(i / 5)].legend(loc=4, fontsize=10)
        plt.show()
    else:
        for lines in range(variogram.shape[2]):
            for i in range(25):
                axs[int(i % 5), int(i / 5)].plot(variogram[:, 0, lines], variogram[:, int((51 - i) * i / 2 + 2), lines],
                                                 linestyle = '--',
                                                 marker = 'x', markersize = 0.5, linewidth = 0.5,
                                                 color = color, label = 'Samples', alpha = 0.8)
                axs[int(i % 5), int(i / 5)].set_xlabel('Distance')
                axs[int(i % 5), int(i / 5)].set_ylabel("%s" % (names[str(i + 1)]), labelpad = 0, size = 20)
        plt.show()


def plot_cross_variogram(variogram):
    var = np.zeros((25, 25))
    for ele in range(25):
        head = int(2 + (25 + 25 - ele) * (ele + 1) / 2 - 25 + ele)
        for j in range(head, head + 25 - ele):
            var[j - head + ele, ele] = np.sum(variogram[:, j] ** 2)
    plt.imshow(var, cmap = 'Reds')
    plt.title('Variance of Cross variogram', size = 15)
    plt.xlabel('Elements')
    plt.ylabel('Elements')
    plt.colorbar(label = 'Sum of square')
    plt.show()


def transport(lm_cdf, if_show=0, show_config=None):
    x = np.random.normal(0, 1, len(lm_cdf[:, 1])).reshape((len(lm_cdf[:, 1]), 1))
    for e in range(len(lm_cdf[1, :]) - 1):
        x = np.hstack((x, np.random.normal(0, 1, len(lm_cdf[:, 1])).reshape((len(lm_cdf[:, 1]), 1))))
    a, b = np.ones((len(lm_cdf),)) / len(lm_cdf), np.ones((len(lm_cdf),)) / len(lm_cdf)
    x_cdf = convert_to_cdf(np.copy(x), if_show = if_show, show_config = show_config, color = 'r')
    dist_matrix = ot.dist(lm_cdf, x_cdf)
    pair = ot.emd(a, b, dist_matrix)
    x_cdf = x_cdf[np.nonzero(pair)[1]]
    x = x[np.nonzero(pair)[1]]
    if if_show == 1:
        plt.plot([lm_cdf[:, show_config[0]], x_cdf[:, show_config[0]]],
                 [lm_cdf[:, show_config[1]], x_cdf[:, show_config[1]]], c = [.5, .5, 1], alpha = 0.2)
        plt.plot(lm_cdf[:, show_config[0]], lm_cdf[:, show_config[1]], '+', c = 'b', label = 'Source samples')
        plt.plot(x_cdf[:, show_config[0]], x_cdf[:, show_config[1]], 'x', c = 'r', label = 'Target samples')
        plt.legend(loc = 0)
        plt.title('OT matrix with samples')
        plt.axis('square')
        plt.show()

    return x, x_cdf


def sgs(data, if_show=0):
    columns = ['X', 'Y', 'Ag', 'Al', 'Au', 'B', 'Ba', 'Be', 'Bi', 'Ca', 'Co', 'F', 'Fe', 'K', 'La', 'Li', 'Mg', 'Mn',
               'Mo', 'Nb', 'P', 'Sn', 'Sr', 'Ti', 'V', 'Y1', 'Zr']
    df = pd.DataFrame(data, columns = columns)
    vario = GSLIB.make_variogram(nug = 0.0, nst = 1, it1 = 1, cc1 = 1.0, azi1 = 0.0, hmaj1 = 50, hmin1 = 50)
    result = np.empty((335 * 335, 25))
    a2g(data, "simulation")
    for i in range(25):
        seed = random.randint(11111, 99999)
        sim = GSLIB.sgsim(1, df, 'X', 'Y', columns[int(i + 2)], 335, 335, 1, seed, vario, "simulation")
        result[:, i] = np.reshape(sim, [335 * 335])
        if i == 10:
            if if_show == 1:
                xmin = 0.0
                xmax = 335.0
                ymin = 0.0
                ymax = 335.0
                cmap = "jet"
                GSLIB.locpix_st(sim, xmin, xmax, ymin, ymax, 1, -4.1, 4.1, df, 'X', 'Y', 'Fe',
                                'Sequential Gaussian Simulation', 'X(km)', 'Y(km)', 'Fe', cmap)
                plt.show()
    return result


class ThinPlateSpline:

    def __init__(self, alpha=0.0) -> None:
        self._fitted = False  # check if fitted
        self.alpha = alpha  #

        self.parameters = np.array([], dtype = np.float32)
        self.control_points = np.array([], dtype = np.float32)

    def fit(self, X: np.ndarray, Y: np.ndarray) -> ThinPlateSpline:
        """Learn f that matches Y given X
        Args:
            X (ndarray): Control point at source space (X_c)
                Shape: (n_c, d_s)
            Y (ndarray): Control point in the target space (X_t)
                Shape: (n_c, d_t)
        """
        assert X.shape[0] == Y.shape[0]

        n_c, d_s = X.shape
        self.control_points = X

        phi = self._radial_distance(X)

        # Build the linear system AP = Y
        X_p = np.hstack([np.ones((n_c, 1)), X])

        A = np.vstack(
            [np.hstack([phi + self.alpha * np.identity(n_c), X_p]), np.hstack([X_p.T, np.zeros((d_s + 1, d_s + 1))])]
        )

        Y = np.vstack([Y, np.zeros((d_s + 1, Y.shape[1]))])

        self.parameters = np.linalg.solve(A, Y)
        self._fitted = True

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        assert self._fitted

        assert X.shape[1] == self.control_points.shape[1]

        phi = self._radial_distance(X)  # n x n_c

        X = np.hstack([phi, np.ones((X.shape[0], 1)), X])  # n x (n_c + 1 + d_s)
        return X @ self.parameters

    def _radial_distance(self, X: np.ndarray) -> np.ndarray:
        dist = cdist(X, self.control_points)
        dist[dist == 0] = 1  # phi(r) = r^2 log(r) ->  (phi(0) = 0)
        return dist ** 2 * np.log(dist)


def de_cdf(anchors, anchors_cdf, data):
    data_decdf = np.zeros(data.shape)
    for ele in range(data.shape[1]):
        i = 0
        rank = np.hstack(
            (anchors[:, ele].reshape((-1, 1)), anchors_cdf[:, ele].reshape((-1, 1))))  # 0-real value 1-cdf value
        rank = rank[rank[:, 1].argsort()]
        bottom = 2*rank[0, 0]-rank[1, 0]
        top = 2*rank[-1, 0]-rank[-2, 0]
        rank = np.vstack(([bottom, 0], rank, [top, 1]))
        data_sorted = np.hstack((data[:, ele].reshape([-1, 1]), np.arange(data.shape[0]).reshape([-1, 1])))
        data_sorted = data_sorted[data_sorted[:, 0].argsort()]
        for idx in range(data.shape[0]):
            if rank[i, 1] <= data_sorted[idx, 0] < rank[i + 1, 1]:
                if data_sorted[idx, 0] == rank[i, 1]:
                    data_decdf[idx, ele] = rank[i, 0]
                else:
                    data_decdf[idx, ele] = (rank[i+1, 0] - rank[i, 0]) / (rank[i+1, 1] - rank[i, 1]) * (data_sorted[idx, 0] - rank[i, 1])+rank[i, 0]
            else:
                i += 1
                data_decdf[idx, ele] = rank[i, 0]
        data_decdf[:, ele] = data_decdf[data_sorted[:, 1].argsort(), ele]
    return data_decdf

def lgt(data, typ):
    mf_logit = np.empty(data.shape)
    for i in range(len(data[1])):
        for idx, x in enumerate(data[:, i]):
            if typ == 1:
                mf_logit[idx, i] = math.log(x / (1 - x))
            else:
                mf_logit[idx, i] = math.exp(x) / (1 + math.exp(x))
    return mf_logit
