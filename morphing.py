import os

from utils import *

if __name__ == '__main__':
    grid = 335
    if_test = False  # test data for 2
    epochs = 1  # simulation times
    nlm = 1122  # number of landmarks
    lag = 4  # lag distance
    nlag = 50  # number of lags in variogram
    mf_repeat = 30  # epoch of simulation
    k = 30  # k nearest neighbor
    if_add = False  # adding points into TPS

    'read origin data'
    rawdata, show, y_label, loc = read_data(test=False, sel=False)

    'shuffle data'
    data = rawdata.copy()
    np.random.shuffle(data)

    'calculate variogram for rawdata'
    if os.path.exists('./variogram_exhausted.npy'):
        rawdata_variogram = np.load('variogram_exhausted.npy')
    else:
        rawdata_variogram = variogram_gam(rawdata, lag=4, nlag=50, trace=True)
        np.save('variogram_exhausted.npy', rawdata_variogram)
    model = vmodel(rawdata_variogram, guess=[0, 10, 110, 0.85])
    plot_variogram([rawdata_variogram], ele=len(y_label), y_label=y_label, line_label=['exhausted data'], colors=['b'],
                   alphas=[1], title='variogram of exhausted data', vmodel=None)

    'creat containers'
    sim_container = np.empty((nlm, data.shape[1], epochs))  # loc+value
    mf_variogram = np.empty((nlag, sum(range(1, data.shape[1] - 1)) + 2, epochs))
    sim_variogram = np.empty((nlag, sum(range(1, data.shape[1] - 1)) + 2, mf_repeat))
    result_container = np.empty((grid * grid, data.shape[1], mf_repeat))  # simulation result container
    result_variogram = np.empty((nlag, sum(range(1, data.shape[1] - 1)) + 2, mf_repeat))

    for epoch in range(epochs):

        "random landmarks"
        landmarks = data[int(epoch * nlm):int(epoch * nlm + nlm), :]
        landmarks[:, 2:] = preprocessing.scale(landmarks[:, 2:])

        'Cumulative distribution of landmarks'
        landmarks_cdf = convert_to_cdf(landmarks.copy(), show_config=show, if_show=True, color='b')

        'Generating MFs'
        mf_raw_container = np.zeros((landmarks.shape[0], landmarks.shape[1], mf_repeat))
        mf_cdf_container = mf_raw_container.copy()
        mf_variogram_container = np.zeros((nlag + 2, sum(range(1, data.shape[1] - 1)) + 2, mf_repeat))
        for r in tqdm(range(mf_repeat), position=0, leave=True):
            if_show = False
            if r <= 2:
                if_show = True
            mf_raw, mf_cdf = transport(landmarks_cdf.copy(), if_show=if_show, show_config=show)
            mf_raw_container[:, :, r] = mf_raw
            mf_cdf_container[:, :, r] = mf_cdf
        mf_ave = np.sum(mf_raw_container, axis=2) / mf_repeat
        plt.scatter(landmarks_cdf[0, show[0]], landmarks_cdf[0, show[1]], marker='+', c='b', label='landmark point')
        plt.scatter(mf_cdf_container[0, show[0], :], mf_cdf_container[0, show[1], :], marker='x', c='r',
                    label='morphing factors')
        plt.legend()
        plt.title('Pairing of landmark point and morphing factors')
        plt.axis('square')
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.show()

        'Sequential gaussian simulation'
        print("\nsimulating")
        for r in tqdm(range(mf_repeat), position=0, leave=False):
            if_show = False
            if r <= 2:
                if_show = True
            mf_sim = sgs(mf_raw_container[:, :, r].copy(), if_show=if_show, vmodel=model)
            result_container[:, :, r] = mf_sim
            'calculate variogram of SGSim'
            sim_variogram[:, :, r] = variogram_gam(mf_sim, lag=4, nlag=50, trace=True)
            result = TPS(mf_sim, mf_raw_container[:, :, r].copy(), landmarks, landmarks_cdf, rawdata
                         , knn=k, if_show=if_show, show=show, add=if_add)
            result_container[:, :, r] = result.copy()
            result_variogram[:, :, r] = variogram_gam(result, lag=4, nlag=50, trace=True)
        plot_variogram([result_variogram, rawdata_variogram], y_label=y_label,
                       line_label=['SMMT', 'rawdata'], colors=['r', 'b'],
                       alphas=[0.5, 1], title='Variogram of SMMT result',
                       vmodel=None)

    'Check result'
    e_type = np.mean(result_container[:, 2:, :], axis=2).reshape((grid, grid, int(rawdata.shape[1] - 2)))
    plt.imshow(e_type[:, :, 0], cmap='jet', origin='lower')
    plt.title("E-type")
    plt.colorbar()
    plt.show()
    std_map = np.std(result_container[:, 2:, :], axis=2).reshape((grid, grid, int(rawdata.shape[1] - 2)))
    plt.imshow(std_map[:, :, 0], cmap='jet', origin='lower')
    plt.title("STD")
    plt.colorbar()
    plt.show()
    pass
