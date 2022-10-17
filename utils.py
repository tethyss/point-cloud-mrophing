from __future__ import annotations
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from scipy.spatial.distance import cdist  # type: ignore
from scipy.optimize import curve_fit
import random
import ot
import subprocess as sp
import math


def read_data(plot=1):
    full_data = pd.read_csv('./data.csv', header=0)
    full_data = full_data.values
    full_data[:, 2:27] = preprocessing.scale(full_data[:, 2:27])
    deposits = pd.read_csv('./deposit.csv', header=0)
    deposits = deposits.values
    deposits = full_data[(335 - deposits[:, 1].astype(int)) * 335 + deposits[:, 0].astype(int) - 1]
    # generate random point
    nongranite = np.argwhere(full_data[:, 29] == 0)
    nongranite = np.reshape(full_data[nongranite], [len(nongranite), 30])
    rand = np.random.randint(len(nongranite), size=49)
    nongranite = nongranite[rand]
    rand = np.random.randint(len(full_data), size=432)
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
    order = np.arange(landmark_points.shape[0])
    np.random.shuffle(order)
    landmark_points = landmark_points[order, :]
    return full_data[:, :27], landmark_points[:500, :27]


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
    sp.run("gam.exe gam.par", stdout=sp.DEVNULL)

    lag = np.arange(1, int(nlag + 1), dtype=float).reshape((nlag, 1))
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


def variogram_calculation(data, lag, steps, tol, channels):
    dist_matrix = ot.dist(data[:, :2], data[:, :2], metric='euclidean')
    'variogram structure: 0: Average distance; 1: counts in lag.'
    variogram = np.zeros((steps, int(2 + channels * (channels + 1) / 2)))
    for i_Step in range(steps):
        print("i_Step :", i_Step + 1)
        x_pos, y_pos = np.where((dist_matrix > lag * (i_Step + 1) - tol) & (dist_matrix < lag * (i_Step + 1) + tol))
        for j in range(len(x_pos)):
            x_j, y_j = x_pos[j], y_pos[j]
            variogram[i_Step, 1] += 1
            variogram[i_Step, 0] += dist_matrix[x_j, y_j]
            dif_j = data[x_j, 2:] - data[y_j, 2:]
            pos = 2
            for c in range(channels):
                variogram[i_Step, int(pos):int(pos + len(dif_j[c:]))] += dif_j[c] * dif_j[c:]
                pos += len(dif_j[c:])
        variogram[i_Step, 2:] = variogram[i_Step, 2:] / (2 * variogram[i_Step, 1])
        variogram[i_Step, 0] = variogram[i_Step, 0] / variogram[i_Step, 1]  # Average distance
        variogram[i_Step, 1] = variogram[i_Step, 1] / 2  # Without duplicates
    return variogram


def variogram_config(variogram):
    nug = np.empty(variogram.shape[1] - 2)

    def func(x, a, b, c):
        return a * x ** 2 + b * x + c

    for dim in range(2, sum(range(1, variogram.shape[1] + 1)) + 2):
        popt, _ = curve_fit(func, variogram[:4, dim], variogram[:4, dim])
        nug[dim - 2] = popt[2]
    return nug


def a2g(rawdata, file):
    columns = ['X', 'Y', 'Ag', 'Al', 'Au', 'B', 'Ba', 'Be', 'Bi', 'Ca', 'Co', 'F', 'Fe', 'K', 'La', 'Li', 'Mg',
               'Mn', 'Mo', 'Nb', 'P', 'Sn', 'Sr', 'Ti', 'V', 'Y1', 'Zr']
    title = np.asarray(file)
    col = np.asarray(columns).reshape((-1, 1))
    head = np.vstack((title, len(columns), col))
    head = pd.DataFrame(head)
    rawdata = pd.DataFrame(rawdata)
    head.to_csv(file, index=False, header=False)
    rawdata.to_csv(file, index=False, header=False, mode="a", sep=' ')


def convert_to_cdf(pre_cdf, show_config, if_show=0, color='b'):  # '#F9E855'
    p = 1. * np.arange(len(pre_cdf) + 2) / (len(pre_cdf) + 1)
    after_cdf = np.empty(pre_cdf.shape)
    after_cdf[:, :2] = pre_cdf[:, :2].copy()
    for ele in range(2, len(pre_cdf[1])):
        data_sorted = pre_cdf[:, ele].copy()
        data_sorted = np.hstack(
            (data_sorted.reshape([-1, 1]), np.arange(len(pre_cdf)).reshape([-1, 1])))
        idex = np.argsort(data_sorted, axis=0)
        data_sorted = data_sorted[idex[:, 0]]
        data_sorted[:, [0, 1]] = data_sorted[:, [1, 0]]
        data_sorted = np.hstack((data_sorted, p[1:len(pre_cdf) + 1].reshape([-1, 1])))
        idex = np.argsort(data_sorted, axis=0)
        data_sorted = data_sorted[idex[:, 0]]
        after_cdf[:, ele] = data_sorted[:, 2]
    if if_show == 1:
        ax1 = plt.subplot(121)
        ax1.set_title('raw data')
        ax1.scatter(pre_cdf[:, show_config[0]], pre_cdf[:, show_config[1]], s=10, c=color)
        ax1.axis('square')
        ax2 = plt.subplot(122)
        ax2.set_title('CDF')
        ax2.scatter(after_cdf[:, show_config[0]], after_cdf[:, show_config[1]], s=10, c=color)
        plt.axis('square')
        plt.show()
    return after_cdf


def plot_variogram(variogram, color="green"):
    names = {'1': 'Ag', '2': 'Al', '3': 'Au', '4': 'B',
             '5': 'Ba', '6': 'Be', '7': 'Bi', '8': 'Ca',
             '9': 'Co', '10': 'F', '11': 'Fe', '12': 'K',
             '13': 'La', '14': 'Li', '15': 'Mg', '16': 'Mn',
             '17': 'Mo', '18': 'Nb', '19': 'P', '20': 'Sn',
             '21': 'Sr', '22': 'Ti', '23': 'V', '24': 'Y',
             '25': 'Zr'}
    fig, axs = plt.subplots(5, 5, figsize=(17, 14))
    plt.suptitle('Direct variogram', size=20)
    plt.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95, wspace=0.5, hspace=0.3)
    for col in range(5):
        for row in range(5):
            axs[row, col].set_ylim(0.0, 1.2)
    if len(variogram.shape) == 2:
        for i in range(25):
            axs[int(i % 5), int(i / 5)].plot(variogram[:, 0], variogram[:, int((51 - i) * i / 2 + 2)], linestyle='--',
                                             marker='x', markersize=0.5, linewidth=0.8,
                                             color=color, label='Samples', )
            axs[int(i % 5), int(i / 5)].set_xlabel('Distance')
            axs[int(i % 5), int(i / 5)].set_ylabel("%s" % (names[str(i + 1)]), labelpad=0, size=20)
            # axs[int(i % 5), int(i / 5)].legend(loc=4, fontsize=10)
    else:
        for lines in range(variogram.shape[2]):
            for i in range(25):
                axs[int(i % 5), int(i / 5)].plot(variogram[:, 0, lines], variogram[:, int((51 - i) * i / 2 + 2), lines],
                                                 linestyle='--',
                                                 marker='x', markersize=0.5, linewidth=0.5,
                                                 color=color, label='Samples', alpha=0.8)
                axs[int(i % 5), int(i / 5)].set_xlabel('Distance')
                axs[int(i % 5), int(i / 5)].set_ylabel("%s" % (names[str(i + 1)]), labelpad=0, size=20)


def plot_cross_variogram(variogram):
    var = np.zeros((25, 25))
    for ele in range(25):
        head = int(2 + (25 + 25 - ele) * (ele + 1) / 2 - 25 + ele)
        for j in range(head, head + 25 - ele):
            var[j - head + ele, ele] = np.sum(variogram[:, j] ** 2)
    plt.imshow(var, cmap='Reds')
    plt.title('Variance of Cross variogram', size=15)
    plt.xlabel('Elements')
    plt.ylabel('Elements')
    plt.colorbar(label='Sum of square')
    plt.show()


def transport(lm_cdf, if_show=False, show_config=None):
    x = lm_cdf.copy()
    for e in range(2, lm_cdf.shape[1]):
        x[:, e] = np.random.normal(0, 1, lm_cdf.shape[0])
    a, b = np.ones((len(lm_cdf),)) / len(lm_cdf), np.ones((len(lm_cdf),)) / len(lm_cdf)
    x_cdf = convert_to_cdf(x.copy(), show_config=show_config, if_show=if_show, color='r')
    dist_matrix = ot.dist(lm_cdf[:, 2:], x_cdf[:, 2:])
    pair = ot.emd(a, b, dist_matrix)
    x_cdf = x_cdf[np.nonzero(pair)[1]]
    x = x[np.nonzero(pair)[1]]
    if if_show:
        plt.plot([lm_cdf[:, show_config[0]], x_cdf[:, show_config[0]]],
                 [lm_cdf[:, show_config[1]], x_cdf[:, show_config[1]]], c=[.5, .5, 1], alpha=0.2)
        plt.plot(lm_cdf[:, show_config[0]], lm_cdf[:, show_config[1]], '+', c='b', label='Source samples')
        plt.plot(x_cdf[:, show_config[0]], x_cdf[:, show_config[1]], 'x', c='r', label='Target samples')
        plt.legend(loc=0)
        plt.title('OT matrix with samples')
        plt.axis('square')
        plt.show()
    return x, x_cdf


def sgs(input, if_show):
    dim = input.shape[1]
    result = np.zeros((335 * 335, dim))
    result[:, :2] = np.hstack(
        (np.tile(np.arange(1, 336), 335).reshape((-1, 1)), np.repeat(np.arange(1, 336), 335).reshape((-1, 1))))
    a2g(input, "data4sim.dat")
    for i in range(2, dim):
        seed = random.randint(11111, 99999)
        cellsize = 1
        nug = 0.0
        nst = 1
        it1 = 1
        cc1 = 1 - nug
        azi1 = 0.0
        max_range = 50
        searchrad = int(max_range / cellsize) * 2 + 1

        with open("sgsim.par", "w") as f:
            f.write("              Parameters for SGSIM                                         \n")
            f.write("              ********************                                         \n")
            f.write("                                                                           \n")
            f.write("START OF PARAMETER:                                                        \n")
            f.write("data4sim.dat                  -file with data                              \n")
            f.write("1  2  0 " + str(i + 1) + "  0  0 -  columns for X,Y,Z,vr,wt,sec.var.          \n")
            f.write("-1.0e21 1.0e21                -  trimming limits                           \n")
            f.write("1                             -transform the data (0=no, 1=yes)            \n")
            f.write("none.trn                      -  file for output trans table               \n")
            f.write("0                             -  consider ref. dist (0=no, 1=yes)          \n")
            f.write("none.dat                      -  file with ref. dist distribution          \n")
            f.write("3 0                           -  columns for vr and wt                     \n")
            f.write("-4.1 4.1                      -zmin,zmax(tail extrapolation)               \n")
            f.write("1   -4.1                      -  lower tail option, parameter              \n")
            f.write("1   4.1                       -  upper tail option, parameter              \n")
            f.write("1                             -debugging level: 0,1,2,3                    \n")
            f.write("debug.dbg                     -file for debugging output                   \n")
            f.write("sgsout.out                    -file for simulation output                  \n")
            f.write("1                             -number of realizations to generate          \n")
            f.write("335 0.5 " + str(cellsize) + " - nx xmn xsiz                                \n")
            f.write("335 0.5 " + str(cellsize) + " - ny ymn ysiz                                \n")
            f.write("1 0.0 1.0                     - nz zmn zsiz                                \n")
            f.write(str(seed) + "                  -random number seed                          \n")
            f.write("0     12                      -min and max original data for sim           \n")
            f.write("18                            -number of simulated nodes to use            \n")
            f.write("0                             -assign data to nodes (0=no, 1=yes)          \n")
            f.write("1     3                       -multiple grid search (0=no, 1=yes),num      \n")
            f.write("0                             -maximum data per octant (0=not used)        \n")
            f.write(str(max_range) + " " + str(max_range) + " 1.0 -maximum search  (hmax,hmin,vert) \n")
            f.write(str(azi1) + "   0.0   0.0       -angles for search ellipsoid                 \n")
            f.write(str(searchrad) + " " + str(searchrad) + " 1 -size of covariance lookup table\n")
            f.write("1     0.60   1.0              -ktype: 0=SK,1=OK,2=LVM,3=EXDR,4=COLC        \n")
            f.write("none.dat                      -  file with LVM, EXDR, or COLC variable     \n")
            f.write("4                             -  column for secondary variable             \n")
            f.write(str(nst) + " " + str(nug) + "  -nst, nugget effect                          \n")
            f.write(str(it1) + " " + str(cc1) + " " + str(azi1) + " 0.0 0.0 -it,cc,ang1,ang2,ang3\n")
            f.write(" " + str(max_range) + " " + str(max_range) + " 1.0 - a_hmax, a_hmin, a_vert \n")

        sp.run("sgsim.exe sgsim.par", stdout=sp.DEVNULL)
        result[:, i] += np.asarray(pd.read_csv('sgsout.out', header=2)).reshape(112225)
        if if_show:
            c_for_show = 10
            if dim == 2:
                c_for_show = 3
            plt.imshow(result[:, c_for_show].reshape(335, 335), cmap='jet', origin='lower', vmax=4.1, vmin=-4.1)
            plt.title("SGSim result with landmarks")
            plt.colorbar()
            plt.scatter(input[:, 0], input[:, 1], c=input[:, c_for_show], cmap="jet", s=20, edgecolor='0.5', vmax=4.1,
                        vmin=-4.1)
            plt.xlim([0, 335])
            plt.ylim([0, 335])
            plt.show()
    return result


class ThinPlateSpline:

    def __init__(self, alpha=0.0) -> None:
        self._fitted = False  # check if fitted
        self.alpha = alpha  #
        self.parameters = np.array([], dtype=np.float32)
        self.control_points = np.array([], dtype=np.float32)

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
        bottom = 2 * rank[0, 0] - rank[1, 0]
        top = 2 * rank[-1, 0] - rank[-2, 0]
        rank = np.vstack(([bottom, 0], rank, [top, 1]))
        data_sorted = np.hstack((data[:, ele].reshape([-1, 1]), np.arange(data.shape[0]).reshape([-1, 1])))
        data_sorted = data_sorted[data_sorted[:, 0].argsort()]
        for idx in range(data.shape[0]):
            if rank[i, 1] <= data_sorted[idx, 0] < rank[i + 1, 1]:
                if data_sorted[idx, 0] == rank[i, 1]:
                    data_decdf[idx, ele] = rank[i, 0]
                else:
                    data_decdf[idx, ele] = (rank[i + 1, 0] - rank[i, 0]) / (rank[i + 1, 1] - rank[i, 1]) * (
                            data_sorted[idx, 0] - rank[i, 1]) + rank[i, 0]
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
