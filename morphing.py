from utils import *

epochs = 200  # simulation times
show = [10, 15]  # config for show
exhausted_variogram = 1

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
for epoch in range(epochs):
    if epoch % (epochs / 5) == 0:
        show_config = 1
    else:
        show_config = 0
    mf_raw, mf_cdf = transport(np.copy(landmarks_cdf), if_show=show_config, show_config=show)

    'Sequential gaussian simulation'
    mf_raw = np.hstack((landmarks[:, 0:2], mf_raw))  # add location
    mf_exhaust = sgs(mf_raw, if_show=show_config)
    sim_result[:, :, epoch] = mf_exhaust.copy()
    mf_exhaust = np.hstack((loc, mf_exhaust))
    if show_config == 1:
        print('computing variogram')
        mf_gamma = variogram_gam(mf_exhaust, grid=[335, 335], cellsize=1, nlag=100)
        plot_variogram(mf_gamma)
        # plot_cross_variogram(gamma)

'Check result'
e_type = np.mean(sim_result, axis=2).reshape((335, 335, 25))
plt.imshow(e_type[:, :, 10], cmap='Spectral', origin='lower')
plt.scatter(landmarks[:, 0], landmarks[:, 1], c='k', s=8)
plt.show()
std_map = np.std(sim_result, axis=2).reshape((335, 335, 25))
plt.imshow(std_map[:, :, 10], cmap='Spectral', origin='lower')
plt.scatter(landmarks[:, 0], landmarks[:, 1], c='k', s=8)
plt.show()

'Match with real data'
# landmarks_exhaust_cdf = tps(mf_exhaust_cdf, mf_cdf, landmarks_cdf)
'Convert back into real values'
# value_sim = de_cdf(landmarks_exhaust_cdf)


# #  logit
# mf_logit = np.empty(mf_cdf.shape)
# for i in range(len(mf_cdf[1])):
#     for idx, x in enumerate(mf_cdf[:, i]):
#         if x != 1 and x != 0:
#             mf_logit[idx, i] = math.log(x / (1 - x))
# plt.figure()
# plt.scatter(mf_logit[:, show[0]], mf_logit[:, show[1]], s=10, c='b')
# plt.title('logit space')
# plt.axis('square')
# plt.show()


# check correlation
# mf_ave = mf_ave / epochs
# mf_variogram = np.hstack((landmarks[:, 0:2], mf_ave))
# dist = ot.dist(landmarks[:, 0:2], landmarks[:, 0:2], metric="euclidean")
# FnMat = variogram_calculation(mf_variogram, dist, lag=4, steps=25, tol=4, channels=25)
# plot_variogram(FnMat, color="red")
pass
