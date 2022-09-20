from utils import *

epochs = 2  # simulation times
show = [10, 15]  # config for show
exhausted_variogram = 1

# read data
data, landmarks = read_data(plot=0)

# calculate exhausted variogram
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

# convert landmarks to CDF
landmarks_cdf = convert_to_cdf(np.copy(landmarks[:, 2:]), show_config=show, if_show=1)

# generating MFs
mf_ave = np.zeros(landmarks_cdf.shape)
for epoch in range(epochs):
    if epoch % (epochs / 2) == 0:
        show_config = 1
    else:
        show_config = 0
    mf_raw, mf_cdf = transport(np.copy(landmarks_cdf), if_show=show_config, show_config=show)
    mf_ave += mf_cdf

    # Sequential gaussian simulation
    mf_raw = np.hstack((landmarks[:, 0:2], mf_raw))  # add location
    mf_exhaust = sgs(mf_raw, if_show=show_config)
    mf_exhaust = np.hstack((data[:, 0:2], mf_exhaust))
    #  logit
    mf_logit = np.empty(mf_exhaust.shape)
    for i in range(len(mf_exhaust[1])):
        for idx, x in enumerate(mf_exhaust[:, i]):
            if x != 1 and x != 0:
                mf_logit[idx, i] = math.log(x / (1 - x))
    plt.hist(mf_logit)
    plt.show()
    rawdata_variogram1 = []
    for v1 in range(25):
        for v2 in range(v1, 25):
            lag1, gamma1 = variogram_gam(mf_exhaust, vcol1=int(v1 + 1), vcol2=int(v2 + 1),
                                         grid=[335, 335], cellsize=1, nlag=20)
            rawdata_variogram1.append(gamma1)
    raw_variogram1 = np.hstack((np.reshape(lag1, [len(lag1), -1]), np.reshape(lag1, [len(lag1), -1])))
    raw_variogram1 = np.hstack((raw_variogram1, np.asarray(rawdata_variogram1).T))
    plot_variogram(raw_variogram1)
    plot_cross_variogram(raw_variogram1)
    os.remove("gam.dat")

    # # Match with real data
    # landmarks_exhaust_cdf = tps(mf_exhaust_cdf, mf_cdf, landmarks_cdf)
    # # Convert back into real values
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
mf_ave = mf_ave / epochs
mf_variogram = np.hstack((landmarks[:, 0:2], mf_ave))
dist = ot.dist(landmarks[:, 0:2], landmarks[:, 0:2], metric="euclidean")
FnMat = variogram_calculation(mf_variogram, dist, lag=4, steps=50, tol=4, channels=25)
plot_variogram(FnMat)
plot_cross_variogram(FnMat)

plt.plot([landmarks_cdf[:, show[0]], mf_ave[:, show[0]]], [landmarks_cdf[:, show[1]], mf_ave[:, show[1]]],
         c=[.5, .5, 1],
         alpha=0.2)
plt.plot(landmarks_cdf[:, show[0]], landmarks_cdf[:, show[1]], '+', c='b', label='Source samples')
plt.plot(mf_ave[:, show[0]], mf_ave[:, show[1]], 'x', c='r', label='Target samples')
plt.legend(loc=0)
plt.title('OT matrix with samples-avg')
plt.axis('square')
plt.show()
