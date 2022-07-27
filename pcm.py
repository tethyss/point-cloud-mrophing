from utils import *
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    DictNames = {'1': 'Ag', '2': 'Al', '3': 'Au', '4': 'B',
                 '5': 'Ba', '6': 'Be', '7': 'Bi', '8': 'Ca',
                 '9': 'Co', '10': 'F', '11': 'Fe', '12': 'K',
                 '13': 'La', '14': 'Li', '15': 'Mg', '16': 'Mn',
                 '17': 'Mo', '18': 'Nb', '19': 'P', '20': 'Sn',
                 '21': 'Sr', '22': 'Ti', '23': 'V', '24': 'Y',
                 '25': 'Zr'}
    show = 1
    data, landmarks = read_data(plot=0)
    landmarks_cdf = convert_to_cdf(np.copy(landmarks))
    # FnMat = exhaust_variogram(data[:,2:])
    print("Computing distances")
    M_Dist = np.zeros([112225, 112225])
    for i in range(335):
        M_Dist[:, int(335 * i):int(335 * (i + 1))] = ot.dist(data[:, 0:2], data[int(335 * i):int(335 * (i + 1)), 0:2],
                                                             metric="euclidean")
        print(i)
    print("Computing variogram")
    FnMat = variogram_calculation(data, M_Dist, lag=4, steps=20, tol=4, channels=25)
    np.savetxt(fname="./VarExp.txt", X=FnMat, fmt='%.4f', delimiter='\t')
    # plot_variogram(FnMat)
    # plot_cross_variogram(FnMat)
    # mf = np.zeros((200, 27))
    # mf[:, 0:2] = landmarks[:, 0:2]
    # for epoch in range(1000):
    #     if epoch % 200 == 0:
    #         show = 1
    #     else:
    #         show = 0
    #     mf[:, 2:27] += transport(np.copy(landmarks[:, 2:]))
    # mf[:, 2:27] = mf[:, 2:27] / 1000
    # mf_cdf = convert_to_cdf(np.copy(mf))
    # plt.plot([landmarks_cdf[:, 12], mf_cdf[:, 12]], [landmarks_cdf[:, 17], mf_cdf[:, 17]], c=[.5, .5, 1], alpha=0.2)
    # plt.plot(landmarks_cdf[:, 12], landmarks_cdf[:, 17], '+b', label='Source samples')
    # plt.plot(mf_cdf[:, 12], mf_cdf[:, 17], 'xr', label='Target samples')
    # plt.legend(loc=0)
    # plt.title('OT matrix with samples-avg')
    # plt.axis('square')
    # plt.show()
    # # M_Dist = ot.dist(mf_cdf[:, 0:2], mf_cdf[:, 0:2], metric="euclidean")
    # # V_mf = variogram_calculation(mf, M_Dist, Lag=4, NSteps=20, LagTol=4, NumVar=25)
    # # np.savetxt(fname="./VarExp_mf.txt", X=V_mf, fmt='%.4f', delimiter='\t')
    # # plot_cross_variogram(V_mf)
