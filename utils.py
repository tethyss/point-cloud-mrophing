from __future__ import annotations
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from scipy.spatial.distance import cdist  # type: ignore
import random
import ot
import subprocess as sp
import math
from scipy import interpolate


def read_data(test):
    """read data
        :arg test:True for tset set
        :return data, show config
    """
    rawdata = pd.read_csv('./data.csv', header = 0)
    rawdata = rawdata.values
    rawdata = rawdata[np.lexsort((rawdata[:, 0], rawdata[:, 1])), :]
    if test:
        return rawdata[:, [0, 1, 12, 17]], [2, 3]
    else:
        return rawdata[:, :27], [12, 17]


def variogram_gam(data, cellsize, nlag):
    a2g(data, "gam.dat")
    num_vario = sum(range(1, data.shape[1] - 1))
    with open("gam.par", "w") as f:
        f.write("                         Parameters for GAM                                  \n")
        f.write("                         *******************                                 \n")
        f.write("                                                                             \n")
        f.write("START OF PARAMETERS:                                                         \n")
        f.write("gam.dat                                 -file with data                      \n")
        if data.shape[1] == 4:
            f.write("2 3 4                               -number of var.,col numbers          \n")
        else:
            f.write(
                "25 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 -number of var.,col numbers\n")
        f.write("-1.0e21     1.0e21                      -trimming limits                     \n")
        f.write("gam_out.out                             -file for variogram output           \n")
        f.write("1                                       -grid or realization number          \n")
        f.write("335   0.5    1                          -nx, xmn, xsiz                       \n")
        f.write("335   0.5    1                          -ny, ymn, ysiz                       \n")
        f.write("1 0 0                                   -nz, zmn, zsiz                       \n")
        f.write("1 " + str(nlag) + "                     -number of directions, number of lags\n")
        f.write(str(cellsize) + "  0  0                  -ixd(1),iyd(1),izd(1)                \n")
        f.write("0                                       -standardize sill? (0=no, 1=yes)     \n")
        f.write(str(num_vario) + "                       -number of variograms                \n")
        for v1 in range(1, data.shape[1] - 1):
            for v2 in range(v1, data.shape[1] - 1):
                f.write(str(v1) + " " + str(v2) + " 2      -tail, head, variogram type  \n")
    # sp.run("gam.exe gam.par", stdout = sp.DEVNULL)
    sp.run("gam.exe gam.par")
    gamma = g2a('gam_out.out', nlag = nlag, num_vario = num_vario)
    return gamma


def variogram_gamv(data, cellsize, nlag, azm, atol, dbglevel=1):
    a2g(data, "gamv.dat")
    num_vario = sum(range(1, data.shape[1] - 1))
    with open("gamv.par", "w") as f:
        f.write("                         Parameters for GAMV                                 \n")
        f.write("                         *******************                                 \n")
        f.write("                                                                             \n")
        f.write("START OF PARAMETERS:                                                         \n")
        f.write("gamv.dat                                -file with data                      \n")
        f.write("1   2   0                               -columns for X, Y, Z coordinates     \n")
        if data.shape[1] == 27:
            f.write("25 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 -number of var.,col num\n")
        else:
            f.write("2 3 4                               -number of var.,col numbers          \n")
        f.write("-1.0e21     1.0e21                      -trimming limits                     \n")
        f.write("gamv_out.out                            -file for variogram output           \n")
        f.write(str(nlag) + "                            -number of lags                      \n")
        f.write(str(cellsize) + "                        -lag separation distance             \n")
        f.write(str(cellsize / 2) + "                    -lag tolerance                       \n")
        f.write("1                                       -number of directions                \n")
        f.write(str(azm) + " " + str(atol) + " 999 0.0 180 999.0 -azm,atol,bandh,dip,dtol,bandv \n")
        f.write("0                                       -standardize sills? (0=no, 1=yes)    \n")
        f.write(str(num_vario) + "                       -number of variograms                \n")
        for v1 in range(1, data.shape[1] - 1):
            for v2 in range(v1, data.shape[1] - 1):
                f.write(str(v1) + " " + str(v2) + " 2    -tail, head, variogram type          \n")
    if dbglevel == 0:
        sp.run("gamv.exe gamv.par", stdout = sp.DEVNULL)
    else:
        sp.run("gamv.exe gamv.par")

    gamma = g2a('gamv_out.out', nlag = nlag, num_vario = num_vario)
    return gamma


def variogram_config(variogram):
    nug = np.zeros(variogram.shape[1] - 2)
    max_r = np.zeros(nug.shape) + 50

    for dim in range(2, variogram.shape[1]):
        nug[dim - 2] = max((2 * variogram[1, dim] - variogram[2, dim]), 0)
        if max(variogram[:, dim]) >= 1:
            'reach'
            max_r[dim - 2] = variogram[np.where(variogram[:, dim] >= 1)[0][0], 0]
        else:
            print('ceil auto generated ')
            'auto'
            i = 0
            a = 1
            while a > 0.001 and i < variogram.shape[0] - 5:
                c = np.polyfit(variogram[i:(i + 5), 0], variogram[i:(i + 5), dim], deg = 1)
                a = c[0]
                i += 1
            max_r[dim - 2] = i + 5
        # 'manual'
        # max_r[dim - 2] = input("range:")

    return np.hstack((nug.reshape([-1, 1]), max_r.reshape([-1, 1])))


def a2g(rawdata, file):
    columns = ['X', 'Y', 'Fe', 'Mn']
    if rawdata.shape[1] == 27:
        columns = ['X', 'Y', 'Ag', 'Al', 'Au', 'B', 'Ba', 'Be', 'Bi', 'Ca', 'Co', 'F', 'Fe', 'K', 'La', 'Li', 'Mg',
                   'Mn', 'Mo', 'Nb', 'P', 'Sn', 'Sr', 'Ti', 'V', 'Y1', 'Zr']
    title = np.asarray(file)
    col = np.asarray(columns).reshape((-1, 1))
    head = np.vstack((title, len(columns), col))
    head = pd.DataFrame(head)
    rawdata = pd.DataFrame(rawdata)
    head.to_csv(file, index = False, header = False)
    rawdata.to_csv(file, index = False, header = False, mode = "a", sep = ' ')


def g2a(file, nlag, num_vario):
    data = []
    gamma = np.zeros((nlag + 2, num_vario + 2))
    with open(file, newline = '') as f:
        for line in f:
            data.append(line.split())
    n = 1
    for i in range(len(data)):
        if i % (nlag + 3) == 0:
            n += 1
        else:
            gamma[i % (nlag + 3) - 1, 0] = data[i][1]
            gamma[i % (nlag + 3) - 1, 1] = data[i][3]
            gamma[i % (nlag + 3) - 1, n] = data[i][2]
    return gamma


def create_vmodel(vmodel, lag, nlag):
    vmodels = np.zeros((nlag+2, vmodel.shape[0]+2))
    for idx, v in enumerate(vmodel):
        with open("vmodel.par", "w") as f:
            f.write("                         Parameters for VMODEL                                  \n")
            f.write("                         *********************                                  \n")
            f.write("START OF PARAMETERS:                                                            \n")
            f.write("vmodel.out                                  -file for variogram output          \n")
            f.write("1 " + str(nlag) + "                         -number of directions and lags      \n")
            f.write("0.0   0.0   " + str(lag) + "                -azm, dip, lag distance             \n")
            f.write("1 " + str(v[0]) + "                         -nst, nugget effect                 \n")
            f.write("1    1  0.0   0.0   0.0                     -it,cc,ang1,ang2,ang3               \n")
            f.write(str(v[1]) + " " + str(v[1]) + " 0.0          -a_hmax, a_hmin, a_vert             \n")
        sp.run("vmodel.exe vmodel.par")
        vmodels[:, idx+2] = g2a('vmodel.out', nlag, vmodel.shape[0])[:, 2]
        vmodels[:, :2] = g2a('vmodel.out', nlag, vmodel.shape[0])[:, :2]
    return vmodels


def convert_to_cdf(pre_cdf, show_config, if_show, color='b'):  # '#F9E855'
    p = 1. * np.arange(len(pre_cdf) + 2) / (len(pre_cdf) + 1)
    after_cdf = np.empty(pre_cdf.shape)
    after_cdf[:, :2] = pre_cdf[:, :2].copy()
    for ele in range(2, len(pre_cdf[1])):
        data_sorted = pre_cdf[:, ele].copy()
        data_sorted = np.hstack(
            (data_sorted.reshape([-1, 1]), np.arange(len(pre_cdf)).reshape([-1, 1])))
        idex = np.argsort(data_sorted, axis = 0)
        data_sorted = data_sorted[idex[:, 0]]
        data_sorted[:, [0, 1]] = data_sorted[:, [1, 0]]
        data_sorted = np.hstack((data_sorted, p[1:len(pre_cdf) + 1].reshape([-1, 1])))
        idex = np.argsort(data_sorted, axis = 0)
        data_sorted = data_sorted[idex[:, 0]]
        after_cdf[:, ele] = data_sorted[:, 2]
    if if_show:
        ax1 = plt.subplot(121)
        ax1.set_title('raw data')
        ax1.scatter(pre_cdf[:, show_config[0]], pre_cdf[:, show_config[1]], s = 10, c = color)
        ax1.axis('square')
        ax2 = plt.subplot(122)
        ax2.set_title('CDF')
        ax2.scatter(after_cdf[:, show_config[0]], after_cdf[:, show_config[1]], s = 10, c = color)
        plt.axis('square')
        plt.show()
    return after_cdf


def plot_variogram(variogram, names, name=None, color="green", vmodel=None):
    # names = {'1': 'Ag', '2': 'Al', '3': 'Au', '4': 'B',
    #          '5': 'Ba', '6': 'Be', '7': 'Bi', '8': 'Ca',
    #          '9': 'Co', '10': 'F', '11': 'Fe', '12': 'K',
    #          '13': 'La', '14': 'Li', '15': 'Mg', '16': 'Mn',
    #          '17': 'Mo', '18': 'Nb', '19': 'P', '20': 'Sn',
    #          '21': 'Sr', '22': 'Ti', '23': 'V', '24': 'Y',
    #          '25': 'Zr'}

    if variogram.shape[1] == 5:
        fig, axs = plt.subplots(1, 3, figsize = (9.5, 3.4))
        plt.suptitle(name, size = 13)
        plt.subplots_adjust(left = 0.1, top = 0.95, bottom = 0.05, right = 0.95, wspace = 0.4)
        if len(variogram.shape) == 2:
            for v in range(2, variogram.shape[1]):
                axs[v - 2].plot(variogram[:, 0], variogram[:, v], linewidth = 0.8, color = color)
                axs[v - 2].set_xlabel('Distance')
                axs[v - 2].set_ylabel(names[v - 2])
                axs[v - 2].set_box_aspect(1)
                axs[v - 2].set_xlim(0.0, max(variogram[:, 0]))
                axs[v - 2].set_ylim(0.0, 1.5)
            axs[1].set_ylim(-0.75, 0.75)
        else:
            for lines in range(variogram.shape[2]):
                for v in range(2, variogram.shape[1]):
                    axs[v - 2].plot(variogram[:, 0, lines], variogram[:, v, lines],
                                    linewidth = 0.8, color = color, alpha=0.2)
                    axs[v - 2].set_xlabel('Distance')
                    axs[v - 2].set_ylabel(names[v - 2])
                    axs[v - 2].set_box_aspect(1)
                    axs[v - 2].set_xlim(0.0, max(variogram[:, 0, lines]))
                    axs[v - 2].set_ylim(0.0, 1.5)
            axs[1].set_ylim(-0.75, 0.75)
    else:
        fig, axs = plt.subplots(5, 5, figsize = (17, 14))
        plt.suptitle(name, size = 20)
        plt.subplots_adjust(left = 0.05, bottom = 0.05, right = 0.95, top = 0.95, wspace = 0.5, hspace = 0.3)
        # for col in range(5):
        #     for row in range(5):
        #         axs[row, col].set_ylim(0.0, 3)
        if len(variogram.shape) == 2:
            for i in range(25):
                axs[int(i % 5), int(i / 5)].plot(variogram[:, 0], variogram[:, int((51 - i) * i / 2 + 2)],
                                                 linestyle = '--',
                                                 marker = 'x', markersize = 0.5, linewidth = 0.8,
                                                 color = color, label = 'Samples')
                axs[int(i % 5), int(i / 5)].set_xlabel('Distance')
                axs[int(i % 5), int(i / 5)].set_ylabel("%s" % (names[str(i + 1)]), labelpad = 0, size = 20)
                # axs[int(i % 5), int(i / 5)].legend(loc=4, fontsize=10)
        else:
            for lines in range(variogram.shape[2]):
                for i in range(25):
                    axs[int(i % 5), int(i / 5)].plot(variogram[:, 0, lines],
                                                     variogram[:, int((51 - i) * i / 2 + 2), lines],
                                                     linestyle = '--',
                                                     marker = 'x', markersize = 0.5, linewidth = 0.5,
                                                     color = color, label = 'Samples', alpha = 0.8)
                    axs[int(i % 5), int(i / 5)].set_xlabel('Distance')
                    axs[int(i % 5), int(i / 5)].set_ylabel("%s" % (names[str(i + 1)]), labelpad = 0, size = 20)
    if vmodel is not None:
        if vmodel.shape[0] == 2:
            axs[0].axvline(vmodel[0, 1], linewidth = 0.8, color = 'k')
            axs[1].axvline(vmodel[1, 1], linewidth = 0.8, color = 'k')
        else:
            for i in range(25):
                axs[int(i % 5), int(i / 5)].axvline(vmodel[int((51 - i) * i / 2), 1], linewidth = 0.8, color = 'k',
                                                    label = 'Samples')
    if name is not None:
        plt.savefig('./variogram of data/' + name + '.png')
    plt.show()


def plot_variogram1(variograms, names, name=None):
    fig, axs = plt.subplots(1, 3, figsize = (9.5, 3.4))
    plt.suptitle(name, size = 13)
    plt.subplots_adjust(left = 0.1, top = 0.95, bottom = 0.05, right = 0.95, wspace = 0.4)
    color = ['orange', 'r', 'k']
    alpha = [0.2, 1, 1]
    for idx, variogram in enumerate(variograms):
        variogram = variogram.reshape((variogram.shape[0], variogram.shape[1],-1))
        for lines in range(variogram.shape[2]):
            for v in range(2, variogram.shape[1]):
                axs[v - 2].plot(variogram[:, 0, lines], variogram[:, v, lines],
                                linewidth = 0.8, color = color[idx], alpha = alpha[idx])
                axs[v - 2].set_xlabel('Distance')
                axs[v - 2].set_ylabel(names[v - 2])
                axs[v - 2].set_box_aspect(1)
                axs[v - 2].set_xlim(0.0, max(variogram[:, 0, lines]))
                axs[v - 2].set_ylim(0.0, 1.5)
        axs[1].set_ylim(-0.75, 0.75)
    if name is not None:
        plt.savefig('./variogram of data/' + name + '.png')
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


def transport(lm_cdf, if_show=False, show_config=None):
    x = lm_cdf.copy()
    for e in range(2, lm_cdf.shape[1]):
        x[:, e] = np.random.normal(0, 1, lm_cdf.shape[0])
    a, b = np.ones((len(lm_cdf),)) / len(lm_cdf), np.ones((len(lm_cdf),)) / len(lm_cdf)
    x_cdf = convert_to_cdf(x.copy(), show_config = show_config, if_show = False, color = 'r')
    dist_matrix = ot.dist(lm_cdf[:, 2:], x_cdf[:, 2:])
    pair = ot.emd(a, b, dist_matrix)
    x_cdf[:, 2:] = x_cdf[np.nonzero(pair)[1], 2:]
    x[:, 2:] = x[np.nonzero(pair)[1], 2:]
    if if_show:
        plt.plot([lm_cdf[:, show_config[0]], x_cdf[:, show_config[0]]],
                 [lm_cdf[:, show_config[1]], x_cdf[:, show_config[1]]], c = [.5, .5, 1], alpha = 0.2)
        plt.plot(lm_cdf[:, show_config[0]], lm_cdf[:, show_config[1]], '+', c = 'b', label = 'Source samples')
        plt.plot(x_cdf[:, show_config[0]], x_cdf[:, show_config[1]], 'x', c = 'r', label = 'Target samples')
        plt.legend(loc = 0)
        plt.title('OT matrix with samples')
        plt.axis('square')
        plt.show()
    return x, x_cdf


def sgs(input, if_show, vmodel):
    dim = input.shape[1]
    result = np.zeros((335 * 335, dim))
    result[:, :2] = np.hstack(
        (np.tile(np.arange(1, 336), 335).reshape((-1, 1)), np.repeat(np.arange(1, 336), 335).reshape((-1, 1))))
    a2g(input, "data4sim.dat")
    for i in range(2, dim):
        seed = random.randint(11111, 99999)
        # nug = vmodel[int((2*(dim-2) + 1 - (i - 2)) * (i - 2) / 2), 0]
        nug = 0
        it1 = 1
        cc1 = 1
        azi1 = 90.0
        v_range = vmodel[int((2 * (dim - 2) + 1 - (i - 2)) * (i - 2) / 2), 1]

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
            f.write("335 0.5 1                     - nx xmn xsiz                                \n")
            f.write("335 0.5 1                     - ny ymn ysiz                                \n")
            f.write("1 0.0 1.0                     - nz zmn zsiz                                \n")
            f.write(str(seed) + "                  -random number seed                          \n")
            f.write("0     24                      -min and max original data for sim           \n")
            f.write("36                            -number of simulated nodes to use            \n")
            f.write("0                             -assign data to nodes (0=no, 1=yes)          \n")
            f.write("1     3                       -multiple grid search (0=no, 1=yes),num      \n")
            f.write("0                             -maximum data per octant (0=not used)        \n")
            f.write("50 50 1.0                     -maximum search  (hmax,hmin,vert)            \n")
            f.write(str(azi1) + "   0.0   0.0      -angles for search ellipsoid                 \n")
            f.write("101 101 1                     -size of covariance lookup table             \n")
            f.write("1     0.60   1.0              -ktype: 0=SK,1=OK,2=LVM,3=EXDR,4=COLC        \n")
            f.write("none.dat                      -  file with LVM, EXDR, or COLC variable     \n")
            f.write("4                             -  column for secondary variable             \n")
            f.write("1 " + str(nug) + "            -nst, nugget effect                          \n")
            f.write(str(it1) + " " + str(cc1) + " " + str(azi1) + " 0.0 0.0 -it,cc,ang1,ang2,ang3\n")
            f.write(str(v_range) + " " + str(v_range) + " 1.0 - a_hmax, a_hmin, a_vert \n")

        # sp.run("sgsim.exe sgsim.par", stdout = sp.DEVNULL)
        sp.run("sgsim.exe sgsim.par")
        result[:, i] += np.asarray(pd.read_csv('sgsout.out', header = 2)).reshape(112225)
    if if_show:
        c_for_show = 10
        if dim == 4:
            c_for_show = 2
        plt.imshow(result[:, c_for_show].reshape(335, 335), cmap = 'jet', origin = 'lower', vmax = 4.1, vmin = -4.1)
        plt.title("SGSim result with landmarks")
        plt.colorbar()
        plt.scatter(input[:, 0], input[:, 1], c = input[:, c_for_show], cmap = "jet", s = 20, edgecolor = '0.5',
                    vmax = 4.1,
                    vmin = -4.1)
        plt.xlim([0, 335])
        plt.ylim([0, 335])
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
        rank = np.hstack(
            (anchors[:, ele].reshape((-1, 1)), anchors_cdf[:, ele].reshape((-1, 1))))  # 0-real value 1-cdf value
        rank = rank[rank[:, 1].argsort()]
        bottom = 2 * rank[0, 0] - rank[1, 0]
        top = 2 * rank[-1, 0] - rank[-2, 0]
        rank = np.vstack(([bottom, 0], rank, [top, 1]))
        func1 = interpolate.interp1d(rank[:, 1], rank[:, 0], kind = 'linear')
        data_decdf[:, ele] = func1(data[:, ele])
    return data_decdf


def lgt(data, typ):
    mf_logit = data.copy()
    for i in range(2, len(data[1])):
        for idx, x in enumerate(data[:, i]):
            if typ == 1:
                mf_logit[idx, i] = math.log(x / (1 - x))
            else:
                mf_logit[idx, i] = math.exp(x) / (1 + math.exp(x))
    return mf_logit
