from utils import *

epochs = 1000
channel = [10, 15]

data, landmarks = read_data(plot=0)
landmarks_cdf = convert_to_cdf(np.copy(landmarks[:, 2:]), show_config=channel, if_show=1)
mf = np.zeros(landmarks_cdf.shape)
for epoch in range(epochs):
    if epoch % (epochs/2) == 0:
        mf += transport(np.copy(landmarks_cdf), if_show=1, show_config=channel)
    else:
        mf += transport(np.copy(landmarks_cdf), if_show=0, show_config=channel)
mf = mf / epochs
plt.plot([landmarks_cdf[:, channel[0]], mf[:, channel[0]]], [landmarks_cdf[:, channel[1]], mf[:, channel[1]]], c=[.5, .5, 1], alpha=0.2)
plt.plot(landmarks_cdf[:, channel[0]], landmarks_cdf[:, channel[1]], '+', c='b', label='Source samples')
plt.plot(mf[:, channel[0]], mf[:, channel[1]], 'x', c='r', label='Target samples')
plt.legend(loc=0)
plt.title('OT matrix with samples-avg')
plt.axis('square')
plt.show()
