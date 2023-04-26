from utils import *
if __name__ == '__main__':
    y_label = ['Ag', 'Al', 'Au', 'B', 'Ba', 'Be', 'Bi', 'Ca', 'Co', 'F', 'Fe', 'K',
               'La', 'Li', 'Mg', 'Mn', 'Mo', 'Nb', 'P', 'Sn', 'Sr', 'Ti', 'V', 'Y1', 'Zr']

    for landmark_portion in [0.01, 0.05]:
        landmarks = np.load('./result/lm_' + str(landmark_portion) + '.npy')
        mf_raw_container = np.load('./result/mf_' + str(landmark_portion) + '.npy')
        model = np.load('./result/vm_' + str(landmark_portion) + '.npy')

        result_container = np.empty((335 * 335, 27, 50))  # simulation result container
        result_variogram = np.empty((50, 327, 50))
        for r in tqdm(range(50), position = 0, leave = True):
            if_show = False
            if r <= 2:
                if_show = True
            mf_sim = sgs(mf_raw_container[:, :, r].copy(), if_show = if_show, vmodel = model)
            result_container[:, :, r] = mf_sim.copy()
            result_variogram[:, :, r] = variogram_gam(mf_sim, lag = 4, nlag = 50, trace = False)
        plot_variogram([result_variogram], ele = len(y_label), y_label = y_label,
                        line_label = ['SGSim'], colors = ['r'],
                        alphas = [0.5], title = 'Variogram of SGSim result', vmodel = model)
        e_type = np.mean(result_container[:, 2:, :], axis = 2).reshape((335, 335, 25))
        plt.imshow(e_type[:, :, 10], cmap = 'jet', origin = 'lower')
        plt.title("E-type")
        plt.colorbar()
        plt.show()
        std_map = np.std(result_container[:, 2:, :], axis = 2).reshape((335, 335, 25))
        plt.imshow(std_map[:, :, 10], cmap = 'jet', origin = 'lower')
        plt.title("STD")
        plt.colorbar()
        plt.show()
        np.save('./result/sgsim_' + str(landmark_portion) + '.npy', result_container)
