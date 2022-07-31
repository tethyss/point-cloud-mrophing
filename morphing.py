import matplotlib.pyplot as plt

from utils import *

epochs = 1000
show = [10, 15]

data, landmarks = read_data(plot=0)
landmarks_cdf = convert_to_cdf(np.copy(landmarks[:, 2:]), show_config=show, if_show=1)
mf = np.zeros(landmarks_cdf.shape)
for epoch in range(epochs):
    if epoch % (epochs/2) == 0:
        mf_cdf = transport(np.copy(landmarks_cdf), if_show=1, show_config=show)
        mf += mf_cdf
    else:
        mf_cdf = transport(np.copy(landmarks_cdf), if_show=0, show_config=show)
        mf += mf_cdf
    mf_logit = np.empty(mf_cdf.shape)
    for i in range(len(landmarks_cdf[1])):
        for idx, x in enumerate(landmarks_cdf[:, i]):
            mf_logit[idx, i] = math.exp(x)/(1+math.exp(x))
    plt.figure()
    plt.scatter(mf_logit[:, show[0]], mf_logit[:, show[1]])
    plt.axis('square')
    plt.show()
mf = mf / epochs
plt.plot([landmarks_cdf[:, show[0]], mf[:, show[0]]], [landmarks_cdf[:, show[1]], mf[:, show[1]]], c=[.5, .5, 1], alpha=0.2)
plt.plot(landmarks_cdf[:, show[0]], landmarks_cdf[:, show[1]], '+', c='b', label='Source samples')
plt.plot(mf[:, show[0]], mf[:, show[1]], 'x', c='r', label='Target samples')
plt.legend(loc=0)
plt.title('OT matrix with samples-avg')
plt.axis('square')
plt.show()
