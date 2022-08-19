from utils import *

epochs = 2  # simulation times
show = [10, 15]  # config for show

# read data
data, landmarks = read_data(plot=0)

# # calculate exhausted variogram
# dist = ot.dist(data[:, 0:2], data[:, 0:2], metric="euclidean")
# FnMat = variogram_calculation(data, dist, lag=4, steps=50, tol=4, channels=25)
# plot_variogram(FnMat)

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
    # # Sequential gaussian simulation
    # mf_raw = np.hstack((landmarks[:, 0:2], mf_raw))  # add location
    # mf_exhaust = Gsim(mf_raw)
    # # Convert into CDF
    # mf_exhaust_cdf = convert_to_cdf(np.copy(mf_exhaust[:, 2:]), show_config=show, if_show=0)
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


plt.plot([landmarks_cdf[:, show[0]], mf_ave[:, show[0]]], [landmarks_cdf[:, show[1]], mf_ave[:, show[1]]], c=[.5, .5, 1],
         alpha=0.2)
plt.plot(landmarks_cdf[:, show[0]], landmarks_cdf[:, show[1]], '+', c='b', label='Source samples')
plt.plot(mf_ave[:, show[0]], mf_ave[:, show[1]], 'x', c='r', label='Target samples')
plt.legend(loc=0)
plt.title('OT matrix with samples-avg')
plt.axis('square')
plt.show()


