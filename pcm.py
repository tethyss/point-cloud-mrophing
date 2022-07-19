import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
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
        plt.legend(fontsize=20, )
        plt.show()
    landmark_points = np.vstack((deposits, nongranite, random_point))
    return full_data[:, :27], landmark_points[:, :27]


def variogram_calculation(BM, M_D, Lag, NSteps, LagTol, NumVar):
    # print(BM)
    FnMat = np.zeros((NSteps + 1, int(2 + NumVar * (NumVar + 1) / 2)))
    for i_Step in range(NSteps):
        print("i_Step :", i_Step + 1)
        tXx, tYy = np.where((M_D > Lag * (i_Step + 1) - LagTol) & (M_D < Lag * (i_Step + 1) + LagTol))
        for jjj in range(len(tXx)):
            i_Sam, j_Sam = tXx[jjj], tYy[jjj]
            FnMat[i_Step + 1, 1] += 1
            FnMat[i_Step + 1, 0] += M_D[i_Sam, j_Sam]
            tPos = 2
            for v1 in range(NumVar):
                for v2 in range(v1, NumVar):
                    FnMat[i_Step + 1, tPos] += (BM[i_Sam, v1 + 2] - BM[j_Sam, v1 + 2]) * (
                            BM[i_Sam, v2 + 2] - BM[j_Sam, v2 + 2])
                    tPos += 1
        FnMat[i_Step + 1, 2:] = FnMat[i_Step + 1, 2:] / (2 * FnMat[i_Step + 1, 1])
        FnMat[i_Step + 1, 0] = FnMat[i_Step + 1, 0] / FnMat[i_Step + 1, 1]  # Average distance
        FnMat[i_Step + 1, 1] = FnMat[i_Step + 1, 1] / 2  # Without duplicates
    return FnMat


def plot_variogram(variogram):
    fig, axs = plt.subplots(5, 5, figsize=(17, 14))
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
        axs[int(i % 5), int(i / 5)].set_ylabel("%s" % (DictNames[str(i + 1)]), labelpad=0)
        # axs[int(i % 5), int(i / 5)].legend(loc=4, fontsize=10)
    plt.show()


def plot_cross_variogram(variogram):

    fig, axs = plt.subplots(25, 25, figsize=(17, 14))
    plt.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95, wspace=0.5, hspace=0.3)
    for i in range(16):
        axs[int(i % 4), int(i / 4)].plot(variogram[:, 0], variogram[:, i + 8], linestyle='--', marker='x',
                                         markersize=0.5, linewidth=0.8,
                                         color='green', label='Samples')
        axs[int(i % 4), int(i / 4)].set_xlabel('Distance')
    plt.show()


def transport(lm):
    x = np.random.normal(0, 2, len(lm[1]))
    for e in range(len(lm)-1):
        x = np.vstack((x, np.random.normal(0, 2, len(lm[1]))))
    a, b = np.ones((len(lm),)) / len(lm), np.ones((len(lm),)) / len(lm)
    M = ot.dist(lm, x)
    G0 = ot.emd(a, b, M)
    if show == 1:
        plot2d_samples_mat(lm, x, G0, c=[.5, .5, 1])
        plt.plot(lm[:, 7], lm[:, 11], '+b', label='Source samples')
        plt.plot(x[:, 7], x[:, 11], 'xr', label='Target samples')
        plt.legend(loc=0)
        plt.title('OT matrix with samples')
        plt.axis('square')
        plt.show()
    x = x[np.nonzero(G0)[1]]
    return x


def plot2d_samples_mat(xs, xt, G, thr=1e-8, **kwargs):
    if ('color' not in kwargs) and ('c' not in kwargs):
        kwargs['color'] = 'k'
    mx = G.max()
    if 'alpha' in kwargs:
        scale = kwargs['alpha']
        del kwargs['alpha']
    else:
        scale = 1
    for i in range(xs.shape[0]):
        for j in range(xt.shape[0]):
            if G[i, j] / mx > thr:
                plt.plot([xs[i, 7], xt[j, 7]], [xs[i, 11], xt[j, 11]],
                         alpha=G[i, j] / mx * scale, **kwargs)


if __name__ == "__main__":
    DictNames = {'1': 'Ag', '2': 'Al', '3': 'Au', '4': 'B',
                 '5': 'Ba', '6': 'Be', '7': 'Bi', '8': 'Ca',
                 '9': 'Co', '10': 'F', '11': 'Fe', '12': 'K',
                 '13': 'La', '14': 'Li', '15': 'Mg', '16': 'Mn',
                 '17': 'Mo', '18': 'Nb', '19': 'P', '20': 'Sn',
                 '21': 'Sr', '22': 'Ti', '23': 'V', '24': 'Y',
                 '25': 'Zr'}
    data, landmarks = read_data(plot=0)
    print("Computing distances for exhausted")
    D_Dist = ot.dist(data[:, 0:2].astype(int), data[:, 0:2].astype(int), metric="euclidean")
    print("Computing distances")
    M_Dist = ot.dist(landmarks[:, 0:2], landmarks[:, 0:2], metric="euclidean")
    print("Computing variogram")
    FnMat = variogram_calculation(landmarks, M_Dist, Lag=4, NSteps=50, LagTol=4, NumVar=25)
    np.savetxt(fname="./VarExp.txt", X=FnMat, fmt='%.4f', delimiter='\t')
    plot_variogram(FnMat)
    plot_cross_variogram(FnMat)
    mf = np.zeros((200, 27))
    mf[:, 0:2] = landmarks[:, 0:2]
    for epoch in range(10000):
        if epoch % 2000 == 0:
            show = 1
        else:
            show = 0
        mf[:, 2:27] += transport(landmarks[:, 2:])
    mf = mf / 10000
    plt.plot([landmarks[:, 9], mf[:, 9]], [landmarks[:, 13], mf[:, 13]], c=[.5, .5, 1], alpha=0.2)
    plt.plot(landmarks[:, 9], landmarks[:, 13], '+b', label='Source samples')
    plt.plot(mf[:, 9], mf[:, 13], 'xr', label='Target samples')
    plt.legend(loc=0)
    plt.title('OT matrix with samples')
    plt.axis('square')
    plt.show()
    V_mf = variogram_calculation(mf, M_Dist, Lag=4, NSteps=60, LagTol=4, NumVar=25)
    np.savetxt(fname="./VarExp_mf.txt", X=V_mf, fmt='%.4f', delimiter='\t')
    plot_cross_variogram(V_mf)
