import matplotlib.pyplot as plt

from utils import *
import probscale

if __name__ == '__main__':
    grid = 335
    if_test = 17  # test data for 2/17
    epochs = 1  # simulation times
    nlm = 200  # number of landmarks
    lag = 4  # lag distance
    nlag = 50  # number of lags in variogram
    mf_repeat = 100  # epoch of simulation
    k = 50  # k nearest neighbor
    if_add = False  # adding points into TPS

    'read origin data'
    rawdata, show, y_label = read_data(test = if_test)

    'shuffle data'
    data = rawdata.copy()
    np.random.shuffle(data)

    'calculate variogram for rawdata'
    rawdata[:, 2:] = preprocessing.scale(rawdata[:, 2:])
    rawdata_variogram = variogram_gam(rawdata, cellsize = 4, nlag = 50)

    'creat containers'
    sim_container = np.empty((nlm, data.shape[1], epochs))  # loc+value
    sim_cdf_container = np.empty((nlm, data.shape[1], epochs))
    mf_variogram = np.empty((nlag, sum(range(1, data.shape[1] - 1)) + 2, epochs))
    sim_variogram = np.empty((nlag, sum(range(1, data.shape[1] - 1)) + 2, mf_repeat))
    result_container = np.empty((grid * grid, data.shape[1], mf_repeat))  # simulation result container
    result_cdf_container = np.empty((grid * grid, data.shape[1], epochs))
    variogram = np.empty((nlag, sum(range(1, data.shape[1] - 1)) + 2, epochs))
    result_variogram = np.empty((nlag, sum(range(1, data.shape[1] - 1)) + 2, mf_repeat))

    for epoch in range(epochs):

        "random landmarks"
        landmarks = data[int(epoch * nlm):int(epoch * nlm + nlm), :]
        landmarks[:, 2:] = preprocessing.scale(landmarks[:, 2:])

        'variogram of landmarks'
        lm_variogram = variogram_gamv(landmarks, cellsize = lag, nlag = nlag, azm = 0, atol = 180, dbglevel = 0)
        # plot_cross_variogram(lm_variogram)
        if epoch < 6:
            plot_variogram([lm_variogram], y_label = y_label, line_label = ['landmark points'], colors = ['b'],
                           alphas = [1], title = 'Variogram of landmarks', vmodel = None)

        'Cumulative distribution of landmarks'
        landmarks_cdf = convert_to_cdf(landmarks.copy(), show_config = show, if_show = True, color = 'b')

        'Generating MFs'
        mf_raw_container = np.zeros((landmarks.shape[0], landmarks.shape[1], mf_repeat))
        mf_cdf_container = mf_raw_container.copy()
        mf_variogram_container = np.zeros((nlag + 2, sum(range(1, data.shape[1] - 1)) + 2, mf_repeat))
        for r in tqdm(range(mf_repeat), position = 0, leave = True):
            if_show = False
            if r <= 2:
                if_show = True
            mf_raw, mf_cdf = transport(landmarks_cdf.copy(), if_show = if_show, show_config = show)
            mf_variogram = variogram_gamv(mf_raw, cellsize = lag, nlag = nlag, azm = 0, atol = 180, dbglevel = 0)
            mf_raw_container[:, :, r] = mf_raw
            mf_cdf_container[:, :, r] = mf_cdf
            mf_variogram_container[:, :, r] = mf_variogram
        plt.scatter(landmarks_cdf[0, show[0]], landmarks_cdf[0, show[1]], marker = '+', c = 'b', label = 'landmark point')
        plt.scatter(mf_cdf_container[0, show[0], :], mf_cdf_container[0, show[1], :], marker = 'x', c = 'r', label = 'morphing factors')
        plt.legend()
        plt.title('Pairing of landmark point and morphing factors')
        plt.axis('square')
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.show()
        mf_ave = np.sum(mf_raw_container, axis = 2) / mf_repeat
        variogram_ave = np.sum(mf_variogram_container, axis = 2) / mf_repeat
        mf_variogram = variogram_gamv(mf_ave, cellsize = lag, nlag = nlag, azm = 0, atol = 180, dbglevel = 0)
        model = vmodel(variogram_ave, guess = [0, 10, 110, 0.85])
        plot_variogram([mf_variogram_container, variogram_ave], y_label = y_label,
                       line_label = ['morphing factors', 'average'], colors = ['orange', 'r'],
                       alphas = [0.5, 1], title = 'Variogram of morphing factors', vmodel = model)
        'Sequential gaussian simulation'
        print("\nsimulating")
        for r in tqdm(range(mf_repeat), position = 0, leave = False):
            if_show = False
            if r <= 2:
                if_show = True
            mf_sim = sgs(mf_raw_container[:, :, r].copy(), if_show = if_show, vmodel = model)
            result_container[:, :, r] = mf_sim
            'calculate variogram of SGSim'
            sim_variogram[:, :, r] = variogram_gam(mf_sim, cellsize = lag, nlag = nlag)
            result, result_cdf = TPS(mf_sim, mf_raw_container[:, :, r].copy(), landmarks, landmarks_cdf, rawdata
                                     , knn = k, if_show = if_show, show = show, add = if_add)
            result_container[:, :, r] = result.copy()
            result_variogram[:, :, r] = variogram_gam(result, cellsize = lag, nlag = nlag)
        plot_variogram([sim_variogram, variogram_ave], y_label = y_label,
                       line_label = ['simulation', 'average'], colors = ['orange', 'r'],
                       alphas = [0.5, 1], title = 'Variogram of simulation result',
                       vmodel = model)
        plot_variogram([result_variogram, rawdata_variogram], y_label = y_label,
                       line_label = ['SMMT', 'rawdata'], colors = ['orange', 'r'],
                       alphas = [0.5, 1], title = 'Variogram of SMMT result',
                       vmodel = None)

    'Check result'
    e_type = np.mean(result_container[:, 2:, :], axis = 2).reshape((grid, grid, int(rawdata.shape[1]-2)))
    plt.imshow(e_type[:, :, 0], cmap = 'jet', origin = 'lower')
    plt.title("E-type")
    plt.colorbar()
    plt.show()
    std_map = np.std(result_container[:, 2:, :], axis = 2).reshape((grid, grid, int(rawdata.shape[1]-2)))
    plt.imshow(std_map[:, :, 0], cmap = 'jet', origin = 'lower')
    plt.title("STD")
    plt.colorbar()
    plt.show()
    pass
