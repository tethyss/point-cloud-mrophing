from __future__ import annotations
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import Rbf
from scipy.spatial.distance import cdist  # type: ignore
import random
import ot
import subprocess as sp
import math
from scipy import interpolate
import scipy.stats as stats
from scipy.stats import percentileofscore
from scipy.optimize import curve_fit
from multiprocessing import Pool
from sklearn.neighbors import NearestNeighbors
from sklearn import preprocessing
from tqdm import trange, tqdm


def read_data(test):
    """read data
        :arg test:True for tset set
        :return data, show config,label
    """
    rawdata = pd.read_csv('./data.csv', header = 0)
    rawdata = rawdata.values
    rawdata = rawdata[np.lexsort((rawdata[:, 0], rawdata[:, 1])), :]
    for i in range(rawdata.shape[1] - 2):
        tail = np.median(rawdata[:, 2 + i]) + 5 * np.std(rawdata[:, 2 + i])
        rawdata[rawdata[:, 2 + i] > tail, 2 + i] = tail
    if test:
        return rawdata[:, [0, 1, 10, 12]], [2, 3], ['Fe', 'Fe-Mn', 'Mn']
    else:
        return rawdata[:, :27], [10, 12], ['Ag', 'Al', 'Au', 'B', 'Ba', 'Be', 'Bi', 'Ca', 'Co', 'F', 'Fe', 'K',
                                           'La', 'Li', 'Mg', 'Mn', 'Mo', 'Nb', 'P', 'Sn', 'Sr', 'Ti', 'V', 'Y1', 'Zr']


def variogram_gam(data, lag, nlag, trace=False):
    a2g(data, "gam.dat")
    num_vario = sum(range(1, data.shape[1] - 1))
    numbers = ' '.join(str(i) for i in range(3, data.shape[1] + 1))
    with open("gam.par", "w") as f:
        f.write("                         Parameters for GAM                                  \n")
        f.write("                         *******************                                 \n")
        f.write("                                                                             \n")
        f.write("START OF PARAMETERS:                                                         \n")
        f.write("gam.dat                                 -file with data                      \n")
        f.write(str(data.shape[1] - 2) + ' ' + numbers + "       -number of var.,col numbers  \n")
        f.write("-1.0e21     1.0e21                      -trimming limits                     \n")
        f.write("gam_out.out                             -file for variogram output           \n")
        f.write("1                                       -grid or realization number          \n")
        f.write(str(int(np.sqrt(data.shape[0]))) + "   0.5  1   -nx, xmn, xsiz                \n")
        f.write(str(int(np.sqrt(data.shape[0]))) + "   0.5  1   -ny, ymn, ysiz                \n")
        f.write("1 0 0                                   -nz, zmn, zsiz                       \n")
        f.write("1 " + str(nlag) + "                     -number of directions, number of lags\n")
        f.write(str(lag) + "  0  0                       -ixd(1),iyd(1),izd(1)                \n")
        f.write("0                                       -standardize sill? (0=no, 1=yes)     \n")
        f.write(str(num_vario) + "                       -number of variograms                \n")
        for v1 in range(1, data.shape[1] - 1):
            for v2 in range(v1, data.shape[1] - 1):
                f.write(str(v1) + " " + str(v2) + " 2      -tail, head, variogram type  \n")
    if trace:
        sp.run("gam.exe gam.par")
    else:
        sp.run("gam.exe gam.par", stdout = sp.DEVNULL)
    gamma = g2a('gam_out.out', nlag = nlag, num_vario = num_vario, type = 'gam')
    return gamma


def variogram_gamv(data, cellsize, nlag, azm, atol, dbglevel=1):
    a2g(data, "gamv.dat")
    num_vario = sum(range(1, data.shape[1] - 1))
    numbers = ' '.join(str(i) for i in range(3, data.shape[1] + 1))
    with open("gamv.par", "w") as f:
        f.write("                         Parameters for GAMV                                 \n")
        f.write("                         *******************                                 \n")
        f.write("                                                                             \n")
        f.write("START OF PARAMETERS:                                                         \n")
        f.write("gamv.dat                                -file with data                      \n")
        f.write("1   2   0                               -columns for X, Y, Z coordinates     \n")
        f.write(str(data.shape[1] - 2) + ' ' + numbers + " -number of var.,col numbers          \n")
        f.write("-1.0e21     1.0e21                      -trimming limits                     \n")
        f.write("gamv_out.out                            -file for variogram output           \n")
        f.write(str(nlag) + "                            -number of lags                      \n")
        f.write(str(cellsize) + "                        -lag separation distance             \n")
        f.write(str(cellsize) + "                        -lag tolerance                       \n")
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

    gamma = g2a('gamv_out.out', nlag = nlag, num_vario = num_vario, type = 'gamv')
    return gamma


def a2g(rawdata, file):
    if rawdata.shape[1] == 27:
        columns = ['X', 'Y', 'Ag', 'Al', 'Au', 'B', 'Ba', 'Be', 'Bi', 'Ca', 'Co', 'F', 'Fe', 'K', 'La', 'Li', 'Mg',
                   'Mn', 'Mo', 'Nb', 'P', 'Sn', 'Sr', 'Ti', 'V', 'Y1', 'Zr']
    elif rawdata.shape[1] == 19:
        columns = ['X', 'Y', 'Al', 'Ba', 'Bi', 'Co', 'F', 'Fe', 'K', 'La', 'Li', 'Mn', 'Mo', 'Nb', 'P', 'Sr', 'Ti', 'V',
                   'Y1']
    elif rawdata.shape[1] == 8:
        columns = ['X', 'Y', 'Z1', 'Z2', 'Z3', 'Z4', 'Z5', 'Z6']
    else:
        columns = ['X', 'Y', 'Fe', 'Mn']
    title = np.asarray(file)
    col = np.asarray(columns).reshape((-1, 1))
    head = np.vstack((title, len(columns), col))
    head = pd.DataFrame(head)
    rawdata = pd.DataFrame(rawdata)
    head.to_csv(file, index = False, header = False)
    rawdata.to_csv(file, index = False, header = False, mode = "a", sep = ' ')


def g2a(file, nlag, num_vario, type):
    data = []
    plus = 0
    if type == 'gamv':
        plus = 2
    gamma = np.zeros((nlag + plus, num_vario + 2))
    with open(file, newline = '') as f:
        for line in f:
            data.append(line.split())
    n = 1
    for i in range(len(data)):
        if i % (nlag + plus + 1) == 0:
            n += 1
        else:
            gamma[i % (nlag + plus + 1) - 1, 0] = data[i][1]
            gamma[i % (nlag + plus + 1) - 1, 1] = data[i][3]
            gamma[i % (nlag + plus + 1) - 1, n] = data[i][2]
    return gamma


def convert_to_cdf(pre_cdf, show_config, if_show, color='b'):
    after_cdf = pre_cdf.copy()
    for ele in range(2, pre_cdf.shape[1]):
        after_cdf[:, ele] = [percentileofscore(pre_cdf[:, ele], x) / 100.0 for x in pre_cdf[:, ele]]
    if if_show:
        ax1 = plt.subplot(121)
        ax1.set_title('raw data')
        ax1.scatter(pre_cdf[:200, show_config[0]], pre_cdf[:200, show_config[1]], s = 10, c = color)
        ax1.axis('square')
        ax2 = plt.subplot(122)
        ax2.set_title('CDF')
        ax2.scatter(after_cdf[:200, show_config[0]], after_cdf[:200, show_config[1]], s = 10, c = color)
        plt.axis('square')
        plt.savefig('./result/datacdf.pdf', dpi = 330)
        plt.show()
    return after_cdf


def plot_variogram(variograms, ele, y_label, line_label, colors, alphas, title, vmodel=None):
    n_cols = 5
    n_rows = math.ceil(ele / n_cols)
    """direct variogram"""
    fig, axs = plt.subplots(n_rows, n_cols, figsize = (15, 2 * n_rows))
    plt.suptitle('Direct ' + title, size = 20)
    plt.subplots_adjust(left = 0.08, bottom = 0.05, right = 0.98, top = 0.95, wspace = 0.5, hspace = 0.2)
    for idx, variogram in enumerate(variograms):
        variogram = variogram.reshape((variogram.shape[0], variogram.shape[1], -1))
        for line in range(variogram.shape[2]):
            for i in range(ele):
                axs[int(i / n_cols), int(i % n_cols)].plot(variogram[:, 0, line],
                                                           variogram[:, int((2 * ele + 1 - i) * i / 2 + 2), line],
                                                           linewidth = 1.5, color = colors[idx], alpha = alphas[idx],
                                                           label = line_label[idx] if line == 0 else '')
                axs[int(i / n_cols), int(i % n_cols)].set_xlabel('Distance', size = 15)
                axs[int(i / n_cols), int(i % n_cols)].set_ylabel(y_label[i], size = 15)
                axs[int(i / n_cols), int(i % n_cols)].tick_params(axis = 'both', labelsize = 15)
                axs[int(i / n_cols), int(i % n_cols)].set_box_aspect(1 / 2)
                # axs[int(i / n_cols), int(i % n_cols)].set_xlim(0.0, max(variogram[:, 0, line]))
                axs[int(i / n_cols), int(i % n_cols)].set_ylim(0, 1.5)
                # axs[int(i / n_cols), int(i % n_cols)].legend()
    if vmodel is not None:
        for i in range(ele):
            axs[int(i / n_cols), int(i % n_cols)].plot(variogram[:, 0, 0], exponential_two(variogram[:, 0, 0],
                                                                                            vmodel[i, 0], vmodel[i, 1],
                                                                                            vmodel[i, 2], vmodel[i, 3])
                                                       , c = 'k', label = 'model',
                                                       linewidth = 0.8)
    plt.savefig('./variogram of data/' + title + '-direct.pdf', dpi = 330)
    plt.show()
    'cross variogram'
    # v = list(set(np.arange(2, sum(range(1, ele + 1)) + 2)) - set(
    #     [int((2 * ele + 1 - i) * i / 2 + 2) for i in range(ele)]))
    v = [6, 35, 36, 41, 44, 47, 102, 110, 123, 145, 177, 185, 186, 193, 210, 216, 217, 221, 226, 229, 232, 242,
         246, 295, 316]
    # random.shuffle(v)
    # TODO: e-e plot
    fig, axs = plt.subplots(n_rows, n_cols, figsize = (15, 2 * n_rows))
    plt.suptitle('Cross ' + title, size = 20)
    plt.subplots_adjust(left = 0.08, bottom = 0.05, right = 0.98, top = 0.95, wspace = 0.5, hspace = 0.2)
    for idx, variogram in enumerate(variograms):
        variogram = variogram.reshape((variogram.shape[0], variogram.shape[1], -1))
        for line in range(variogram.shape[2]):
            for i in range(int(n_rows * n_cols)):
                axs[int(i / n_cols), int(i % n_cols)].plot(variogram[:, 0, line], variogram[:, v[i], line],
                                                           linewidth = 1.5, color = colors[idx], alpha = alphas[idx],
                                                           label = line_label[idx] if line == 0 else '')
                axs[int(i / n_cols), int(i % n_cols)].set_xlabel('Distance')
                axs[int(i / n_cols), int(i % n_cols)].set_box_aspect(1 / 2)
                axs[int(i / n_cols), int(i % n_cols)].tick_params(axis = 'both', labelsize = 15)
                # axs[int(i / 5), int(i % 5)].set_xlim(0.0, max(variogram[:, 0, line]))
                axs[int(i / 5), int(i % 5)].set_ylim(-1.2, 1.2)
                # axs[int(i / n_cols), int(i % n_cols)].legend()
    plt.savefig('./variogram of data/' + title + '-cross.pdf', dpi = 330)
    plt.show()


def plot_cross_variogram(variogram):
    if all(variogram.shape[1] == 327):
        e = 25
    else:
        e = 17
    var = np.zeros((e, e))
    for ele in range(e):
        head = int(2 + (2 * e - ele) * (ele + 1) / 2 - e + ele)
        for j in range(head, head + e - ele):
            var[j - head + ele, ele] = np.sum(variogram[:, j] ** 2)
    plt.imshow(var, cmap = 'Reds')
    plt.title('Variance of Cross variogram', size = 15)
    plt.xlabel('Elements')
    plt.ylabel('Elements')
    plt.colorbar(label = 'Sum of square')
    plt.show()


def transport(lm_cdf, if_show=False, show_config=None):
    x = lm_cdf.copy()
    x_cdf = lm_cdf.copy()
    for e in range(2, lm_cdf.shape[1]):
        x[:, e] = np.random.normal(0, 1, lm_cdf.shape[0])
        x_cdf[:, e] = stats.norm.cdf(x[:, e], loc = 0, scale = 1)
    a, b = np.ones((len(lm_cdf),)) / len(lm_cdf), np.ones((len(lm_cdf),)) / len(lm_cdf)
    dist_matrix = ot.dist(lm_cdf[:, 2:], x_cdf[:, 2:])
    pair = ot.emd(a, b, dist_matrix, numItermax=1e8)
    x_cdf[:, 2:] = x_cdf[np.nonzero(pair)[1], 2:]
    x[:, 2:] = x[np.nonzero(pair)[1], 2:]
    if if_show:
        ax1 = plt.subplot(121)
        ax1.set_title('Morphing factors')
        ax1.scatter(x[:200, show_config[0]], x[:200, show_config[1]], s = 10, c = 'r')
        ax1.axis('square')
        ax2 = plt.subplot(122)
        ax2.set_title('CDF')
        ax2.scatter(x_cdf[:200, show_config[0]], x_cdf[:200, show_config[1]], s = 10, c = 'r')
        plt.axis('square')
        plt.savefig('./result/mfcdf.pdf', dpi = 330)
        plt.show()
        plt.plot([lm_cdf[:100, show_config[0]], x_cdf[:100, show_config[0]]],
                 [lm_cdf[:100, show_config[1]], x_cdf[:100, show_config[1]]], c = [.5, .5, 1], alpha = 0.2)
        plt.plot(lm_cdf[:100, show_config[0]], lm_cdf[:100, show_config[1]], '+', c = 'b', label = 'landmark points')
        plt.plot(x_cdf[:100, show_config[0]], x_cdf[:100, show_config[1]], 'x', c = 'r', label = 'morphing factors')
        plt.legend(loc = 0)
        plt.title('OT matrix with samples')
        plt.axis('square')
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.savefig('./result/ot.pdf', dpi=330)
        plt.show()
    return x, x_cdf


def show_connect(lm_cdf, mf_cdf, show_config=[10,12]):
    dis = np.abs(lm_cdf[:, 2:]-0.5)
    row_index = np.argmin(dis[:, show_config[0]-2]+dis[:, show_config[1]-2])
    d = np.sum(np.sqrt((mf_cdf[row_index, show_config[0], :] - lm_cdf[row_index, show_config[0]])**2+\
         (mf_cdf[row_index, show_config[1], :] - lm_cdf[row_index, show_config[1]])**2))
    d = round(d, 2)
    plt.scatter(lm_cdf[row_index, show_config[0]], lm_cdf[row_index, show_config[1]], marker='+', c='b',label='landmark point')
    plt.scatter(mf_cdf[row_index, show_config[0], :], mf_cdf[row_index, show_config[1], :], marker='x', c='r',
                label='morphing factors')
    plt.legend()
    plt.title('Pairing of landmark and morphing factors '+str(d))
    plt.axis('square')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.show()

def sgs(input, if_show, vmodel):
    grid = 335
    if input.shape[0] == 199:
        grid = 200
    dim = input.shape[1]
    result = np.zeros((grid * grid, dim))
    result[:, :2] = np.hstack(
        (np.tile(np.arange(1, grid + 1), grid).reshape((-1, 1)),
         np.repeat(np.arange(1, grid + 1), grid).reshape((-1, 1))))
    a2g(input, "data4sim.dat")
    par = []
    for i in range(2, dim):
        parname = './exe/sgsim' + str(i - 2) + '.par'
        seed = random.randint(11111, 99999)
        nug = vmodel[int(i - 2), 0]
        it1 = 2
        cc1 = vmodel[int(i - 2), 3]
        azi1 = 0.0
        range1 = vmodel[int(i - 2), 1]
        range2 = vmodel[int(i - 2), 2]

        with open(parname, "w") as f:
            f.write("              Parameters for SGSIM                                         \n")
            f.write("              ********************                                         \n")
            f.write("                                                                           \n")
            f.write("START OF PARAMETER:                                                        \n")
            f.write("data4sim.dat                  -file with data                              \n")
            f.write("1  2  0 " + str(i + 1) + "  0  0 -  columns for X,Y,Z,vr,wt,sec.var.       \n")
            f.write("-1.0e21 1.0e21                -  trimming limits                           \n")
            f.write("0                             -transform the data (0=no, 1=yes)            \n")
            f.write("none.trn                      -  file for output trans table               \n")
            f.write("0                             -  consider ref. dist (0=no, 1=yes)          \n")
            f.write("none.dat                      -  file with ref. dist distribution          \n")
            f.write("3 0                           -  columns for vr and wt                     \n")
            f.write("-4.1 4.1                      -zmin,zmax(tail extrapolation)               \n")
            f.write("1   -4.1                      -  lower tail option, parameter              \n")
            f.write("1   4.1                       -  upper tail option, parameter              \n")
            f.write("1                             -debugging level: 0,1,2,3                    \n")
            f.write("debug.dbg                     -file for debugging output                   \n")
            f.write("./exe/sgsout" + str(i - 2) + ".out  -file for simulation output                  \n")
            f.write("1                             -number of realizations to generate          \n")
            f.write(str(grid) + " 0.5 1              - nx xmn xsiz                                \n")
            f.write(str(grid) + " 0.5 1              - ny ymn ysiz                                \n")
            f.write("1 0.0 1.0                     - nz zmn zsiz                                \n")
            f.write(str(seed) + "                  -random number seed                          \n")
            f.write("0     16                      -min and max original data for sim           \n")
            f.write("24                            -number of simulated nodes to use            \n")
            f.write("1                             -assign data to nodes (0=no, 1=yes)          \n")
            f.write("1     3                       -multiple grid search (0=no, 1=yes),num      \n")
            f.write("0                             -maximum data per octant (0=not used)        \n")
            f.write("60 60 10                      -maximum search  (hmax,hmin,vert)            \n")
            f.write(str(azi1) + "   0.0   0.0      -angles for search ellipsoid                 \n")
            f.write("241 241 1                     -size of covariance lookup table             \n")
            f.write("0     0.0   1.0              -ktype: 0=SK,1=OK,2=LVM,3=EXDR,4=COLC        \n")
            f.write("none.dat                      -  file with LVM, EXDR, or COLC variable     \n")
            f.write("4                             -  column for secondary variable             \n")
            f.write("2 " + str(nug) + "            -nst, nugget effect                          \n")
            f.write(str(it1) + " " + str(cc1) + " " + str(azi1) + " 0.0 0.0 -it,cc,ang1,ang2,ang3\n")
            f.write(str(range1) + " " + str(range1) + " 1.0 - a_hmax, a_hmin, a_vert \n")
            f.write(str(it1) + " " + str(1 - nug - cc1) + " " + str(azi1) + " 0.0 0.0 -it,cc,ang1,ang2,ang3\n")
            f.write(str(range2) + " " + str(range2) + " 1.0 - a_hmax, a_hmin, a_vert \n")

        par.append("sgsim.exe " + str(parname))
    with Pool(12) as p:
        p.map(sgsrun, par)
    for i in range(2, dim):
        result[:, i] += np.asarray(pd.read_csv('./exe/sgsout' + str(i - 2) + '.out', header = 2)).reshape(
            grid*grid)
    if if_show:
        c_for_show = 10
        if dim == 4:
            c_for_show = 2
        elif dim == 17:
            c_for_show = 7
        elif dim == 8:
            c_for_show = 2
        plt.imshow(result[:, c_for_show].reshape(grid, grid), cmap = 'jet', origin = 'lower', vmax = 4.1, vmin = -4.1)
        plt.title("SGSim result with landmarks")
        plt.colorbar()
        plt.scatter(input[:, 0], input[:, 1], c = input[:, c_for_show], cmap = "jet", s = 20, edgecolor = '0.5',
                    vmax = 4.1,
                    vmin = -4.1)
        plt.xlim([0, grid])
        plt.ylim([0, grid])
        plt.show()
    return result


def sgsrun(par):
    sp.run(par, stdout = sp.DEVNULL)


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
        func1 = interpolate.interp1d(rank[:, 1], rank[:, 0], kind = 'quadratic')
        data_decdf[:, ele] = func1(data[:, ele])
    return data_decdf


def lgt(data, typ):
    mf_logit = data.copy()
    for i in range(2, len(data[1])):
        for idx, x in enumerate(data[:, i]):
            if typ == 1:
                mf_logit[idx, i] = math.log(x / (1 - x))
            else:
                mf_logit[idx, i] = np.exp(x) / (1 + np.exp(x))
    return mf_logit


def exponential_two(h, nug, r1, r2, c):
    result = np.zeros(h.shape)
    for idx, r in enumerate(h):
        if r <= r1:
            result[idx] = c * (1 - math.exp(-3 * r / r1))
        elif r1 <= r <= r2:
            result[idx] = c + (1 - nug - c) * (1 - math.exp(-3 * (r - r1) / r2))
        else:
            result[idx] = c + (1 - nug - c) * (1 - math.exp(-3 * (r2 - r1) / r2))
    return result + nug


def vmodel(variogram, guess=None):
    if guess is None:
        guess = [0, 10, 110, 0.85]
    x = variogram[:, 0]
    if variogram.shape[1] == 5:
        parameters = np.zeros((2, 4))
    elif variogram.shape[1] == 327:
        parameters = np.zeros((25, 4))
    else:
        parameters = np.zeros((6, 4))
    for i in range(parameters.shape[0]):
        v = int((2 * parameters.shape[0] + 1 - i) * i / 2)
        y = variogram[:, v + 2]
        parameters[i], _ = curve_fit(exponential_two, x, y, p0 = guess, bounds = (0, 335))
    return parameters


def TPS(sim, mf, lm, lm_cdf, rawdata, knn, if_show, show, add=True):
    grid = 335
    mf_cdf = mf.copy()
    sim_cdf = sim.copy()
    for e in range(2, len(mf[1])):
        mf_cdf[:, e] = stats.norm.cdf(mf[:, e], loc = 0, scale = 1)
        sim_cdf[:, e] = stats.norm.cdf(sim[:, e], loc = 0, scale = 1)
    sim_cdf_lgt = lgt(sim_cdf.copy(), typ = 1)
    mf_cdf_lgt = lgt(mf_cdf.copy(), typ = 1)
    lm_cdf_lgt = lgt(lm_cdf.copy(), typ = 1)
    mf_base = mf_cdf_lgt.copy()
    lm_base = lm_cdf_lgt.copy()
    np.random.shuffle(sim_cdf_lgt)
    result_cdf_lgt = sim_cdf_lgt.copy()

    for idx, x in enumerate(sim_cdf_lgt[:, :2]):
        if not max(np.all(lm_base[:, :2] == x, axis = 1)):
            loc = search_box(x, lm_base, knn)
            nbrs = NearestNeighbors(n_neighbors = knn, algorithm = 'auto').fit(lm_base[loc, :2])
            tps = ThinPlateSpline()
            *_, indices = nbrs.kneighbors([x])
            tps.fit(mf_base[loc[indices], 2:].reshape(knn, -1), lm_base[loc[indices], 2:].reshape(knn, -1))
            result_cdf_lgt[idx, 2:] = tps.transform(sim_cdf_lgt[idx, 2:].reshape(1, -1))
            if add:
                mf_base = np.vstack((mf_base, sim_cdf_lgt[idx]))
                lm_base = np.vstack((lm_base, result_cdf_lgt[idx]))
    result_cdf = lgt(result_cdf_lgt.copy(), typ = -1)
    result = de_cdf(lm[:, 2:], lm_cdf[:, 2:], result_cdf[:, 2:])
    result = np.hstack((result_cdf[:, :2], result))
    result = result[np.lexsort((result[:, 0], result[:, 1])), :].copy()
    result_cdf = result_cdf[np.lexsort((result_cdf[:, 0], result_cdf[:, 1])), :].copy()
    if if_show:
        plt.imshow(result[:, show[0]].reshape(grid, grid), cmap = 'jet', origin = 'lower')
        plt.colorbar()
        plt.title("SMMT result")
        plt.show()
        plt.imshow(rawdata[:, show[0]].reshape(grid, grid), cmap = 'jet', origin = 'lower',
                   vmax = np.max(result[:, show[0]]), vmin = np.min(result[:, show[0]]))
        plt.colorbar()
        plt.title("Original data")
        plt.show()
    return result


def search_box(x, pool, knn):
    density = (335 * 335) / len(pool)
    loc = []
    rate = 1
    while len(loc) <= knn:
        r = math.ceil(math.sqrt(knn * density) * rate)
        x0 = max(0, int(x[0] - r))
        x1 = min(335, int(x[0] + r))
        y0 = max(0, int(x[1] - r))
        y1 = min(335, int(x[1] + r))
        loc = np.where((x0 <= pool[:, 0]) & (pool[:, 0] <= x1) & (y0 <= pool[:, 1]) & (pool[:, 1] <= y1))
        loc = np.asarray(loc).reshape(-1)
        rate = rate * 1.05
    return loc
