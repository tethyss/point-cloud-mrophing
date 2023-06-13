from utils import *

def plot_variogram(variograms):
    fig, axs = plt.subplots(1, 3, figsize=(15, 2.5))
    plt.subplots_adjust(left=0.08, bottom=0.05, right=0.98, top=0.95, wspace=0.5, hspace=0.2)
    for i in range(variograms.shape[2]):
        axs[0].plot(variograms[:, 0, i], variograms[:, 2, i], linewidth=1.5, color='b')
        axs[0].tick_params(axis='both', labelsize=15)
        axs[0].set_box_aspect(1 / 2)
        axs[1].plot(variograms[:, 0, i], variograms[:, 3, i], linewidth=1.5, color='b')
        axs[1].tick_params(axis='both', labelsize=15)
        axs[1].set_box_aspect(1 / 2)
        axs[2].plot(variograms[:, 0, i], variograms[:, 4, i], linewidth=1.5, color='b')
        axs[2].tick_params(axis='both', labelsize=15)
        axs[2].set_box_aspect(1 / 2)
    plt.savefig('./variogram of data/variogram of SMMT_test.pdf', dpi=330)
    plt.show()


if __name__ == '__main__':
    k = 18
    landmarks = np.load('./result/lm_test.npy')
    landmarks_cdf = convert_to_cdf(landmarks.copy(), show_config=[2, 3], if_show=True, color='b')
    rawdata = np.load('./result/rawdata_test.npy')
    mf_raw_container = np.load('./result/mf_test.npy')
    sim_result = np.load('./result/sgsim_test.npy')
    result_variogram = np.empty((100, 5, 50))
    result_container = sim_result.copy()
    for r in trange(sim_result.shape[2]):
        if_show = False
        if r <= 2:
            if_show = True
        result = TPS(sim_result[:, :, r].copy(), mf_raw_container[:, :, r].copy(), landmarks.copy(), landmarks_cdf.copy(),
                     rawdata.copy(), knn=k, if_show=if_show, show=[2, 3])
        result_container[:, :, r] = result.copy()
        result_variogram[:, :, r] = variogram_gam(result, lag=2, nlag=100, trace=False)
    np.save('./result/vario_SMMT_test.npy', result_variogram)
    np.save('./result/result_SMMT_test.npy', result_container)