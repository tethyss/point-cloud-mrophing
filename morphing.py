from utils import *

epochs = 10  # simulation times
show = [10, 15]  # config for show
exhausted_variogram = 0

"read data"
data, landmarks = read_data(plot=0)

'calculate exhausted variogram'
if exhausted_variogram == 1:
    rawdata_variogram = []
    for v1 in range(25):
        for v2 in range(v1, 25):
            lag, gamma = variogram_gam(data, vcol1=int(v1 + 1), vcol2=int(v2 + 1),
                                       grid=[335, 335], cellsize=1, nlag=20)
            rawdata_variogram.append(gamma)
    raw_variogram = np.hstack((np.reshape(lag, [len(lag), -1]), np.reshape(lag, [len(lag), -1])))
    raw_variogram = np.hstack((raw_variogram, np.asarray(rawdata_variogram).T))
    plot_variogram(raw_variogram)
    plot_cross_variogram(raw_variogram)
    os.remove("gam.dat")

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
    if os.path.exists("gam.dat"):
        os.remove("gam.dat")
    rawdata_variogram1 = []
    print('computing variogram')
    for v1 in range(25):
        for v2 in range(v1, 25):
            lag1, gamma1 = variogram_gam(mf_exhaust, vcol1=int(v1 + 1), vcol2=int(v2 + 1),
                                         grid=[335, 335], cellsize=1, nlag=100)
            rawdata_variogram1.append(gamma1)
    raw_variogram1 = np.hstack((np.reshape(lag1, [len(lag1), -1]), np.reshape(lag1, [len(lag1), -1])))
    raw_variogram1 = np.hstack((raw_variogram1, np.asarray(rawdata_variogram1).T))
    plot_variogram(raw_variogram1)
    # plot_cross_variogram(raw_variogram1)

'Check result'
#  e_type=
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
