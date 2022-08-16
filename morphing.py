from scipy.interpolate import Rbf
from utils import *

epochs = 1000
show = [10, 15]

data, landmarks = read_data(plot=0)
landmarks_cdf = convert_to_cdf(np.copy(landmarks[:, 2:]), show_config=show, if_show=1)
mf = np.zeros(landmarks_cdf.shape)
for epoch in range(epochs):
    if epoch % (epochs / 2) == 0:
        mf_cdf = transport(np.copy(landmarks_cdf), if_show=1, show_config=show)
        mf += mf_cdf
    else:
        mf_cdf = transport(np.copy(landmarks_cdf), if_show=0, show_config=show)
        mf += mf_cdf

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

mf = mf / epochs
dist = ot.dist(landmarks[:, 0:2], landmarks[:, 0:2], metric="euclidean")
FnMat = variogram_calculation(data, dist, lag = 4, steps = 50, tol = 4, channels = 25)
# plot_cross_variogram(FnMat)


plt.plot([landmarks_cdf[:, show[0]], mf[:, show[0]]], [landmarks_cdf[:, show[1]], mf[:, show[1]]], c=[.5, .5, 1],
         alpha=0.2)
plt.plot(landmarks_cdf[:, show[0]], landmarks_cdf[:, show[1]], '+', c='b', label='Source samples')
plt.plot(mf[:, show[0]], mf[:, show[1]], 'x', c='r', label='Target samples')
plt.legend(loc=0)
plt.title('OT matrix with samples-avg')
plt.axis('square')
plt.show()

mf_cdf = transport(np.copy(landmarks_cdf), if_show=0, show_config=show)
tps_function = Rbf(landmarks[:, 0], landmarks[:, 1], mf_cdf[:, 0], function = 'thin_plate')
exhaust_mf = tps_function(np.linspace(1, 335, 335), np.linspace(1, 335, 335))
