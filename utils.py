import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import geostatspy.GSLIB as GSLIB
import random
import ot
import math


def read_data(plot=1):
    full_data = pd.read_csv('./data.csv', header=0)
    full_data = full_data.values
    full_data[:, 2:27] = StandardScaler().fit_transform(full_data[:, 2:27])
    deposits = pd.read_csv('./deposit.csv', header=0)
    deposits = deposits.values
    deposits = full_data[(deposits[:, 0].astype(int) - 1) * 335 + deposits[:, 1].astype(int)]
    # generate random point
    nongranite = np.argwhere(full_data[:, 29] == 0)
    nongranite = np.reshape(full_data[nongranite], [len(nongranite), 30])
    rand = np.random.randint(len(nongranite), size=49)
    nongranite = nongranite[rand]
    rand = np.random.randint(len(full_data), size=102)
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
    return full_data[:, :27], landmark_points[:, :27]


def exhaust_variogram(rawdata):
    rawdata = np.reshape(rawdata, [335, 335, 25])
    rawdata = rawdata[::-1]
    lag = 2
    steps = 40
    length = int(lag * steps * 2)
    width = int(length / 2)
    dist = np.zeros((width, length))

    for le in range(length):
        for j in range(width):
            dist[j, le] = math.sqrt(j ** 2 + (le + 1 - width) ** 2)

    group = []
    for n in range(steps):
        group.append(np.argwhere((dist >= int(n * lag)) & (dist < ((n + 1) * lag))))
        g = np.zeros((width, length))

    for gr in range(len(group)):
        for point in group[gr]:
            g[int(point[0]), int(point[1])] = gr
    plt.imshow(g, origin='lower')
    plt.show()

    h = np.zeros((len(group), int(25 * 26 / 2)))
    count = np.zeros((len(group)))
    for i in range(112225):  # point
        for j in range(len(group)):  # group
            for ind in group[j]:  # location in group
                if 0 <= int(i / 335) - width + ind[1] < 335 and 0 <= int(i % 335) + ind[0] < 335:
                    p = 0
                    for m in range(25):
                        for n in range(m, 25):
                            h[j, p] += (rawdata[int(i / 335), int(i % 335), m] - rawdata[
                                int(i / 335) - width + ind[1], int(i % 335) + ind[0], m]) * (
                                               rawdata[int(i / 335), int(i % 335), n] - rawdata[
                                           int(i / 335) - width + ind[1], int(i % 335) + ind[0], n])
                            p += 1
                    count[j] += 1
        if i % 1000 == 0:
            print("Done:", i)
    h_p = h / (2 * count.reshape(len(h), 1))
    return h_p


def convert_to_cdf(data1, if_show=0, show_config=None, color='b'):  # '#F9E855'
    if show_config is None:
        show_config = [10, 15]
    if if_show == 1:
        plt.figure(1)
        plt.title('raw data')
        plt.scatter(data1[:, show_config[0]], data1[:, show_config[1]], s=10, c=color)  # '#FF1F5B'
        plt.axis('square')
    p = 1. * np.arange(len(data1)) / (len(data1) - 1)
    for ele in range(len(data1[1])):
        data_sorted = data1[:, ele]
        data_sorted = np.hstack(
            (data_sorted.reshape([len(data1), 1]), np.arange(len(data1)).reshape([len(data1), 1])))
        idex = np.argsort(data_sorted, axis=0)
        data_sorted = data_sorted[idex[:, 0]]
        data_sorted[:, [0, 1]] = data_sorted[:, [1, 0]]
        data_sorted = np.hstack((data_sorted, p.reshape([len(data1), 1])))
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


def plot_variogram(variogram):
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
    # for col in range(5):
    #     for row in range(5):
    #         axs[row, col].set_ylim(0.0, 1)
    #         axs[row, col].set_xlim(0.0, 20)
    #         axs[row, col].set_xticks((0, 5, 10, 15, 20))
    for i in range(25):
        axs[int(i % 5), int(i / 5)].plot(variogram[:, 0], variogram[:, int((51 - i) * i / 2 + 2)], linestyle='--',
                                         marker='x', markersize=0.5, linewidth=0.8,
                                         color='green', label='Samples')
        axs[int(i % 5), int(i / 5)].set_xlabel('Distance')
        axs[int(i % 5), int(i / 5)].set_ylabel("%s" % (names[str(i + 1)]), labelpad=0, size=20)
        # axs[int(i % 5), int(i / 5)].legend(loc=4, fontsize=10)
    plt.show()


def plot_cross_variogram(variogram):
    var = np.zeros((25, 25))
    for ele in range(25):
        head = int(2 + (25 + 25 - ele) * (ele + 1) / 2 - 25 + ele)
        for j in range(head, head + 25 - ele):
            var[j - head + ele, ele] = np.sum(variogram[:, j] ** 2)
    plt.imshow(var, cmap='Reds')
    plt.title('Cross variogram', size=15)
    plt.xlabel('Elements')
    plt.ylabel('Elements')
    plt.colorbar(label='Sum of square')
    plt.show()


def transport(lm_cdf, if_show=0, show_config=None):
    x = np.random.normal(0, 1, len(lm_cdf[1]))
    for e in range(len(lm_cdf) - 1):
        x = np.vstack((x, np.random.normal(0, 1, len(lm_cdf[1]))))
    a, b = np.ones((len(lm_cdf),)) / len(lm_cdf), np.ones((len(lm_cdf),)) / len(lm_cdf)
    x_cdf = convert_to_cdf(np.copy(x), if_show=if_show, show_config=show_config, color='r')
    dist_matrix = ot.dist(lm_cdf, x_cdf)
    pair = ot.emd(a, b, dist_matrix)
    x_cdf = x_cdf[np.nonzero(pair)[1]]
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


def sgs(data, if_show=0):
    columns = ['X', 'Y', 'Ag', 'Al', 'Au', 'B', 'Ba', 'Be', 'Bi', 'Ca', 'Co', 'F', 'Fe', 'K', 'La', 'Li', 'Mg', 'Mn',
               'Mo', 'Nb', 'P', 'Sn', 'Sr', 'Ti', 'V', 'Y1', 'Zr']
    df = pd.DataFrame(data, columns = columns)
    vario = GSLIB.make_variogram(nug = 0.0, nst = 1, it1 = 1, cc1 = 1.0, azi1 = 60.0, hmaj1 = 100000, hmin1 = 100000)
    result = np.empty((335*335, 25))
    for i in range(25):
        seed = random.randint(11111, 99999)
        sim = GSLIB.sgsim(1, df, 'X', 'Y', columns[int(i+2)],335,335,1000,seed,vario,"simulation")
        result[:, i] = np.reshape(sim, [335*335])
        if i == 11:
            if if_show == 1:
                xmin = 0.0
                xmax = 335000.0
                ymin = 0.0
                ymax = 335000.0
                cmap = plt.cm.inferno
                GSLIB.locpix_st(sim, xmin, xmax, ymin, ymax, 1000, 0.0, 1.0, df, 'X', 'Y', 'Fe',
                                'Sequential Gaussian Simulation', 'X(m)', 'Y(m)', 'Fe', cmap)
                plt.show()
    return result
#
#
# def tps(data, mf, landmarks):
#
#     tps_function = Rbf(landmarks[:, 0], landmarks[:, 1], mf_cdf[:, 0], function='thin_plate')
#     exhaust_mf = tps_function(np.linspace(1, 335, 335), np.linspace(1, 335, 335))
#     return data
#
#
# def de_cdf(data):
#     pass
#     return data
