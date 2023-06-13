from utils import *


def plot_variogram(variograms):
    fig, axs = plt.subplots(1, 3, figsize=(15, 2.5))
    plt.subplots_adjust(left=0.08, bottom=0.05, right=0.98, top=0.95, wspace=0.5, hspace=0.2)
    for i in range(variograms.shape[2]):
        axs[0].plot(variograms[:, 0, i], variograms[:, 2, i], linewidth=1.5, color='b')
        axs[0].tick_params(axis='both', labelsize=15)
        axs[0].set_box_aspect(1 / 2)
        axs[0].set_ylim(0, 1.5)
        axs[1].plot(variograms[:, 0, i], variograms[:, 3, i], linewidth=1.5, color='b')
        axs[1].tick_params(axis='both', labelsize=15)
        axs[1].set_box_aspect(1 / 2)
        axs[1].set_ylim(-0.75, 0.75)
        axs[2].plot(variograms[:, 0, i], variograms[:, 4, i], linewidth=1.5, color='b')
        axs[2].tick_params(axis='both', labelsize=15)
        axs[2].set_box_aspect(1 / 2)
        axs[2].set_ylim(0, 1.5)
    plt.savefig('./variogram of data/variogram of sgsim_test.pdf', dpi=330)
    plt.show()


if __name__ == '__main__':
    mf_raw_container = np.load('./result/mf_test.npy')
    rawdata_variogram = np.load('./result/variogram_exhausted_test.npy')
    model = vmodel(rawdata_variogram, guess=[0, 10, 110, 0.85])
    mf_repeat = 50
    result_container = np.empty((335 * 335, 4, mf_repeat))  # simulation result container
    result_variogram = np.empty((100, 5, mf_repeat))

    'Sequential gaussian simulation'
    for r in tqdm(range(mf_repeat), position=0, leave=True):
        if_show = False
        if r <= 2:
            if_show = True
        mf_sim = sgs(mf_raw_container[:, :, r].copy(), if_show=if_show, vmodel=model)
        result_container[:, :, r] = mf_sim.copy()
        result_variogram[:, :, r] = variogram_gam(mf_sim, lag=2, nlag=100, trace=False)
    plot_variogram(result_variogram)
    e_type = np.mean(result_container[:, 2:, :], axis=2).reshape((335, 335, 2))
    plt.imshow(e_type[:, :, 1], cmap='jet', origin='lower')
    plt.title("E-type")
    plt.colorbar()
    plt.show()
    std_map = np.std(result_container[:, 2:, :], axis=2).reshape((335, 335, 2))
    plt.imshow(std_map[:, :, 1], cmap='jet', origin='lower')
    plt.title("STD")
    plt.colorbar()
    plt.show()
    np.save('./result/sgsim_test.npy', result_container)
