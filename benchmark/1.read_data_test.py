import os
from utils import *


if __name__ == '__main__':
    'load origin data'
    if os.path.exists('./result/bmdata.npy'):
        rawdata = np.load('./result/bmdata.npy')
    else:
        rawdata = np.empty((200, 200, 6))
        for i in range(6):
            rawdata[:, :, i] = np.flipud(np.loadtxt('bmdata/Reference_Z' + str(i + 1) + '_numpy.txt'))
        np.save('./result/bmdata.npy', rawdata)


    'load conditioning points'
    cdpoints = np.load('./bmdata/cdpoints.npy')
    plt.imshow(rawdata[:, :, 0], cmap='jet', origin='lower')
    plt.scatter(cdpoints[:, 0], cdpoints[:, 1], s=5, c='k')
    plt.show()

    'calculate variogram'
    data = np.empty((200 * 200, 8))
    for i in range(200 * 200):
        data[i, :] = np.hstack((int(i % 200), int(i // 200), rawdata[i // 200, i % 200, :]))
    M_Dist = ot.dist(cdpoints[:, 0:2], cdpoints[:, 0:2], metric="euclidean")
    variogram_cdpoints = variogram_omni(cdpoints, M_Dist, Lag=4, Nlag=30, LagTol=4, NumVar=6)

    np.save('./result/variogram_cdpoints.npy', variogram_cdpoints)

