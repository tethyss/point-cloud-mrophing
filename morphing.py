from utils import *
import probscale

if __name__ == '__main__':
    if_test = 17  # test data for 2/17
    epochs = 1  # simulation times
    nlm = 1122  # number of landmarks
    lag = 4  # lag distance
    nlag = 50  # number of lags in variogram
    mf_repeat = 10  # epoch of simulaiton
    k = 10  # k nearest neighbor
    if_add = True  # adding points into TPS

    # 'creat locations for simulation result'
    # loc = np.hstack(
    #     (np.tile(np.arange(1, 336), 335).reshape((-1, 1)), np.repeat(np.arange(1, 336), 335).reshape((-1, 1))))

    'read origin data'
    rawdata, show, y_label = read_data(test = 17)

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
    result_container = np.empty((335 * 335, data.shape[1], mf_repeat))  # simulation result container
    result_cdf_container = np.empty((335 * 335, data.shape[1], epochs))
    variogram = np.empty((nlag, sum(range(1, data.shape[1] - 1)) + 2, epochs))
    result_variogram = np.empty((nlag, sum(range(1, data.shape[1] - 1)) + 2, mf_repeat))

    for epoch in range(epochs):
        if_show = False

        "random landmarks"
        landmarks = data[int(epoch * nlm):int(epoch * nlm + nlm), :]
        landmarks[:, 2:] = preprocessing.scale(landmarks[:, 2:])

        'variogram of landmarks'
        lm_variogram = variogram_gamv(landmarks, cellsize = lag, nlag = nlag, azm = 0, atol = 180, dbglevel = 0)
        # plot_cross_variogram(lm_variogram)
        if epoch < 6:
            plot_variogram([lm_variogram], y_label = y_label, line_label = ['landmark'], colors = ['r'],
                           alphas = [1], title = 'variogram of landmarks-epoch ' + str(epoch), vmodel = None)

        'Cumulative distribution of landmarks'
        landmarks_cdf = convert_to_cdf(landmarks.copy(), show_config = show, if_show = False, color = 'b')

        'Generating MFs'
        mf_raw_container = np.zeros((landmarks.shape[0], landmarks.shape[1], mf_repeat))
        mf_cdf_container = mf_raw_container.copy()
        mf_variogram_container = np.zeros((nlag + 2, sum(range(1, data.shape[1] - 1)) + 2, mf_repeat))
        for r in tqdm(range(mf_repeat), position = 0, leave = True):
            if r <= 2:
                if_show = True
            mf_raw, mf_cdf = transport(landmarks_cdf.copy(), if_show = False, show_config = show)
            mf_variogram = variogram_gamv(mf_raw, cellsize = lag, nlag = nlag, azm = 0, atol = 180, dbglevel = 0)
            mf_raw_container[:, :, r] = mf_raw
            mf_cdf_container[:, :, r] = mf_cdf
            mf_variogram_container[:, :, r] = mf_variogram
        if if_show:
            plt.scatter(landmarks_cdf[0, show[0]], landmarks_cdf[0, show[1]], c = 'b', label = 'landmark')
            plt.scatter(mf_cdf_container[0, show[0], :], mf_cdf_container[0, show[1], :], c = 'r', label = 'morphing points')
            plt.legend()
            plt.axis('square')
            plt.xlim(0, 1)
            plt.ylim(0, 1)
            plt.show()
        mf_ave = np.sum(mf_raw_container, axis = 2) / mf_repeat
        variogram_ave = np.sum(mf_variogram_container, axis = 2) / mf_repeat
        mf_variogram = variogram_gamv(mf_ave, cellsize = lag, nlag = nlag, azm = 0, atol = 180, dbglevel = 0)
        model = vmodel(variogram_ave, guess = [0, 10, 110, 0.85])
        plot_variogram([mf_variogram_container, variogram_ave], y_label = y_label,
                       line_label = ['morphing factor', 'average'], colors = ['orange', 'r'],
                       alphas = [0.5, 1], title = 'variogram of morphing factors-epoch ' + str(epoch), vmodel = model)
        'Sequential gaussian simulation'
        print("\nsimulating")
        for r in tqdm(range(mf_repeat), position = 0, leave = True):
            mf_sim = sgs(mf_raw_container[:, :, r].copy(), if_show = if_show, vmodel = model)
            result_container[:, :, r] = mf_sim
            'calculate variogram of SGSim'
            sim_variogram[:, :, r] = variogram_gam(mf_sim, cellsize = lag, nlag = nlag)
            result, result_cdf = TPS(mf_sim, mf_raw_container[:, :, r].copy(), landmarks, landmarks_cdf, rawdata
                                     , knn = k, if_show = if_show, show = show, add = if_add)
            result_container[:, :, r] = result.copy()
            result_variogram[:, :, r] = variogram_gam(result, cellsize = lag, nlag = nlag)
        if if_show:
            plot_variogram([sim_variogram, variogram_ave], y_label = y_label,
                           line_label = ['simulation', 'average'], colors = ['orange', 'r'],
                           alphas = [0.5, 1], title = 'variogram of simulation result-epoch ' + str(epoch),
                           vmodel = model)
            plot_variogram([result_variogram, rawdata_variogram], y_label = y_label,
                           line_label = ['SMMT', 'rawdata'], colors = ['orange', 'r'],
                           alphas = [0.5, 1], title = 'variogram of SMMT result-epoch ' + str(epoch),
                           vmodel = None)

    # 'calculate result variogram'
    #
    # 'show pdf'
    # common_opts = dict(
    #     plottype = 'prob',
    #     probax = 'y',
    #     datascale = 'log',
    #     datalabel = 'Fe',
    #     scatter_kws = dict(c = 'g', marker = '.', linestyle = 'none', markersize = 0.5)
    # )
    # for e in range(epochs):
    #     fig = probscale.probplot(result_container[:, 10, e] + 5, dist = None, **common_opts)
    # common_opts = dict(
    #     plottype = 'prob',
    #     probax = 'y',
    #     datascale = 'log',
    #     datalabel = 'Fe',
    #     scatter_kws = dict(c = 'r', marker = '.', linestyle = 'none', markersize = 0.5)
    # )
    # fig = probscale.probplot(data[:, 12] + 5, dist = None, **common_opts)
    # plt.show()

    'Check result'
    e_type = np.mean(result_container[:, 2:, :], axis = 2).reshape((335, 335, 2))
    plt.imshow(e_type[:, :, 0], cmap = 'jet', origin = 'lower')
    plt.title("E-type")
    plt.colorbar()
    plt.show()
    std_map = np.std(result_container[:, 2:, :], axis = 2).reshape((335, 335, 2))
    plt.imshow(std_map[:, :, 0], cmap = 'jet', origin = 'lower')
    plt.title("STD")
    plt.colorbar()
    plt.show()
    pass
