from utils import *
import time


start = time.time()
epochs = 10  # simulation times
show = [10, 15]  # config for show
exhausted_variogram = 0

"read data"
data, landmarks = read_data(plot=1)

'calculate exhausted variogram'
if exhausted_variogram == 1:
    gamma = variogram_gam(data, grid=[335, 335], cellsize=1, nlag=50)
    plot_variogram(gamma)
    plot_cross_variogram(gamma)

'convert landmarks to CDF'
landmarks_cdf = convert_to_cdf(np.copy(landmarks[:, 2:]), show_config=show, if_show=1)

'creat locations for simulation result'
loc1 = np.tile(np.arange(1, 336, dtype=int), 335)
loc2 = np.repeat(np.arange(1, 336, dtype=int), 335)
loc = np.hstack((loc1.reshape((-1, 1)), loc2.reshape((-1, 1))))

'Generating MFs'
mf_ave = np.zeros(landmarks_cdf.shape)
sim_result = np.empty((335 * 335, 25, epochs))  # simulation result container
cdf_result = np.empty((335 * 335, 25, epochs))
nlag = 50
variogram = np.empty((nlag, 327, epochs))
for epoch in range(epochs):
    # if epoch % (epochs / 4) == 0:
    #     show_config = 1
    # else:
    #     show_config = 0
    show_config = 1
    mf_raw, mf_cdf = transport(np.copy(landmarks_cdf), if_show=show_config, show_config=show)

    'Sequential gaussian simulation'
    mf_raw = np.hstack((landmarks[:, 0:2], mf_raw))  # add location
    mf_exhaust = sgs(mf_raw, if_show=show_config)
    sim_result[:, :, epoch] = mf_exhaust.copy()

    'TPS'
    mf_exhaust_cdf = convert_to_cdf(mf_exhaust.copy())
    m_cdf = lgt(mf_cdf.copy(), typ = 1)
    lm_cdf = lgt(landmarks_cdf.copy(), typ = 1)
    exhaust_cdf = lgt(mf_exhaust_cdf.copy(), typ = 1)
    tps = ThinPlateSpline()
    tps.fit(m_cdf, lm_cdf)
    sim_cdf = tps.transform(exhaust_cdf)
    sim_cdf = lgt(sim_cdf.copy(), typ = -1)
    sim = de_cdf(landmarks[:, 2:], landmarks_cdf, sim_cdf)
    if show_config == 1:
        plt.imshow(np.flipud(sim[:, 10].reshape(335, 335)), cmap = 'jet', origin = 'lower')
        plt.colorbar()
        plt.title("Morphing result")
        plt.show()
        plt.imshow(np.flipud(data[:, 12].reshape(335, 335)), cmap = 'jet', origin = 'lower', vmax = np.max(sim[:,10]),vmin = np.min(sim[:,10]))
        plt.colorbar()
        plt.title("Original data")
        plt.show()
    sim_result[:, :, epoch] = sim.copy()

    # print('computing variogram')
    # result = np.hstack((loc, sim.copy()))
    # mf_gamma = variogram_gam(result, grid=[335, 335], cellsize=1, nlag=nlag)
    # variogram[:, :, epoch] = mf_gamma.copy()
plot_variogram(variogram)

'show pdf'
for e in range(epochs):
    plt.scatter(sim_result[:, 10, e], sim_cdf[:, 10], c = 'y', s = 1)
plt.scatter(landmarks[:, 12], landmarks_cdf[:, 10], c = '0', s = 2)
plt.show()


'Check result'
e_type = np.mean(sim_result, axis = 2).reshape((335, 335, 25))
plt.imshow(np.flipud(e_type[:, :, 10]), cmap = 'jet', origin = 'lower')
plt.title("E-type")
plt.colorbar()
plt.scatter(landmarks[:, 0], landmarks[:, 1], c = 'k', s = 8)
plt.show()
std_map = np.std(sim_result, axis = 2).reshape((335, 335, 25))
plt.imshow(np.flipud(std_map[:, :, 10]), cmap = 'jet', origin = 'lower')
plt.title("STD")
plt.colorbar()
plt.show()

end = time.time()
print((end - start)/60)

pass
