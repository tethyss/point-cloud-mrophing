import numpy as np
from sklearn.preprocessing import StandardScaler

from utils import *

if __name__ == '__main__':
    grid = 200
    if_test = False  # test data for 2
    epochs = 100  # simulation times
    nlm = 199  # number of landmarks
    lag = 4  # lag distance
    nlag = 30  # number of lags in variogram
    mf_repeat = 100  # epoch of simulation
    k = 30  # k nearest neighbor
    if_add = False  # adding points into TPS

    'read origin data'
    rawdata = np.empty((200, 200, 6))
    for i in range(6):
        rawdata[:, :, i] = np.loadtxt('./benchmark/Reference_Z' + str(i + 1) + '_numpy.txt')
    'add location'
    data = np.empty((200 * 200, 8))
    # for i in range(200*200):
    #    data[i,:] = np.hstack((int(i%200), int(i//200), rawdata[i//200, i%200,:]))
    # data[:,2:]=(data[:,2:]-np.mean(data[:,2:], axis=0))/np.std(data[:,2:], axis=0)

    "random landmarks"
    landmarks = np.loadtxt('./benchmark/Conditioning_Data_6dim_Numpy.txt')
    landmarks[:, 2:] = (landmarks[:, 2:] - np.mean(landmarks[:, 2:], axis=0)) / np.std(landmarks[:, 2:], axis=0)

    'visualization'
    fig, axs = plt.subplots(2, 3, figsize=(9.5, 6.4))
    for i in range(6):
        axs[i // 3, i % 3].imshow(rawdata[:, :, i], cmap='jet')
        axs[i // 3, i % 3].scatter(landmarks[:, 0], landmarks[:, 1], c='none', s=10, edgecolor='grey')
        axs[i // 3, i % 3].set_title('Z' + str(i + 1))
    plt.show()

    # 'calculate variogram for rawdata'
    # rawdata_variogram = variogram_gam(data, cellsize=lag, nlag=nlag)
    #
    # 'variogram of landmarks'
    # lm_variogram = variogram_gamv(landmarks, cellsize=lag, nlag=nlag, azm=0, atol=180, dbglevel=0)
    #
    y_label = ['Z1', 'Z2', 'Z3', 'Z4', 'Z5', 'Z6']
    #
    # plot_variogram([rawdata_variogram, lm_variogram], y_label=y_label, line_label=['rawdata', 'landmark points'],
    #                colors=['r', 'b'],
    #                alphas=[1, 1], title='Variogram of rawdata and landmarks', vmodel=None)

    'creat containers'
    sim_container = np.empty((nlm, data.shape[1], epochs))  # loc+value
    sim_cdf_container = np.empty((nlm, data.shape[1], epochs))
    mf_variogram = np.empty((nlag, sum(range(1, data.shape[1] - 1)) + 2, epochs))
    sim_variogram = np.empty((nlag, sum(range(1, data.shape[1] - 1)) + 2, mf_repeat))
    result_container = np.empty((grid * grid, data.shape[1], mf_repeat))  # simulation result container
    result_cdf_container = np.empty((grid * grid, data.shape[1], epochs))
    variogram = np.empty((nlag, sum(range(1, data.shape[1] - 1)) + 2, epochs))
    result_variogram = np.empty((nlag, sum(range(1, data.shape[1] - 1)) + 2, mf_repeat))

    'Cumulative distribution of landmarks'
    landmarks_cdf = convert_to_cdf(landmarks.copy(), show_config=[2, 3], if_show=True, color='b')

    'Generating MFs'
    mf_raw_container = np.zeros((landmarks.shape[0], landmarks.shape[1], mf_repeat))
    mf_cdf_container = mf_raw_container.copy()
    mf_variogram_container = np.zeros((nlag + 2, sum(range(1, data.shape[1] - 1)) + 2, mf_repeat))
    for r in tqdm(range(mf_repeat), position=0, leave=True):
        if_show = False
        if r <= 2:
            if_show = True
        mf_raw, mf_cdf = transport(landmarks_cdf.copy(), if_show=if_show, show_config=[2,3])
        mf_variogram = variogram_gamv(mf_raw, cellsize=lag, nlag=nlag, azm=0, atol=180, dbglevel=0)
        mf_raw_container[:, :, r] = mf_raw
        mf_cdf_container[:, :, r] = mf_cdf
        mf_variogram_container[:, :, r] = mf_variogram
    plt.scatter(landmarks_cdf[0, 2], landmarks_cdf[0, 3], marker='+', c='b', label='landmark point')
    plt.scatter(mf_cdf_container[0, 2, :], mf_cdf_container[0, 3, :], marker='x', c='r',
                label='morphing factors')
    plt.legend()
    plt.title('Pairing of landmark point and morphing factors')
    plt.axis('square')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.show()
    mf_ave = np.sum(mf_raw_container, axis=2) / mf_repeat
    variogram_ave = np.sum(mf_variogram_container, axis=2) / mf_repeat
    mf_variogram = variogram_gamv(mf_ave, cellsize=lag, nlag=nlag, azm=0, atol=180, dbglevel=0)
    model = vmodel(variogram_ave, guess=[0, 10, 110, 0.85])
    plot_variogram([mf_variogram_container, variogram_ave], y_label=y_label,
                   line_label=['morphing factors', 'average'], colors=['orange', 'r'],
                   alphas=[0.5, 1], title='Variogram of morphing factors', vmodel=model)

    'Sequential gaussian simulation'
    print("\nsimulating")
    for r in tqdm(range(mf_repeat), position=0, leave=False):
        if_show = False
        if r <= 2:
            if_show = True
        mf_sim = sgs(mf_raw_container[:, :, r].copy(), if_show=if_show, vmodel=model)
        result_container[:, :, r] = mf_sim
        'calculate variogram of SGSim'
        sim_variogram[:, :, r] = variogram_gam(mf_sim, cellsize=lag, nlag=nlag)
        result, result_cdf = TPS(mf_sim, mf_raw_container[:, :, r].copy(), landmarks, landmarks_cdf, rawdata
                                 , knn=k, if_show=if_show, show=show, add=if_add)
        result_container[:, :, r] = result.copy()
        result_variogram[:, :, r] = variogram_gam(result, cellsize=lag, nlag=nlag)
    plot_variogram([sim_variogram, variogram_ave], y_label=y_label,
                   line_label=['simulation', 'average'], colors=['orange', 'r'],
                   alphas=[0.5, 1], title='Variogram of simulation result',
                   vmodel=model)
    plot_variogram([result_variogram, rawdata_variogram], y_label=y_label,
                   line_label=['SMMT', 'rawdata'], colors=['orange', 'r'],
                   alphas=[0.5, 1], title='Variogram of SMMT result',
                   vmodel=None)
    np.save('result' + str(epoch) + '.npy', result_container)
