from utils import *

'read origin data'
rawdata, show, y_label, loc = read_data(test=False, sel=False)
'shuffle data'
data = rawdata.copy()
data[:, 2:] = preprocessing.scale(data[:, 2:])
data[loc, 2:] = -999
data = data[:, 2:].reshape(335, 335, -1)
gamma = np.zeros((50, 2 + 25))
for i in range(1, 51):
    x = (data[:(-i * 4), :, :] - data[(i * 4):, :, :]) ** 2
    x = x.reshape(-1, 25)
    x = x[np.argwhere(x[:, 0] != 0).reshape(-1)]
    x = x[np.argwhere(x[:, 0] < 666).reshape(-1)]
    x = x[np.argwhere(x[:, 0] > -666).reshape(-1)]
    gamma[i - 1, 0] = i * 4
    gamma[i - 1, 1] = len(x[:, 1])
    gamma[i - 1, 2:] = np.sum(x, axis=0) / (2 * len(x[:, 1]))
fig, axs = plt.subplots(5, 5, figsize=(17, 14))
plt.suptitle('Conditional GAM', size=20)
plt.subplots_adjust(left=0.1, top=0.95, bottom=0.05, right=0.95, wspace=0.4)
for i in range(25):
    axs[int(i / 5), int(i % 5)].plot(gamma[:, 0], gamma[:, 2+i], linewidth=0.8, color='b', alpha=1,label='gam')
    axs[int(i / 5), int(i % 5)].set_xlabel('Distance')
    axs[int(i / 5), int(i % 5)].set_ylabel(y_label[i])
    axs[int(i / 5), int(i % 5)].set_box_aspect(1)
    axs[int(i / 5), int(i % 5)].set_xlim(0.0, max(gamma[:, 0]))
    axs[int(i / 5), int(i % 5)].set_ylim(0.0, 1.5)
    axs[int(i / 5), int(i % 5)].legend()
plt.show()
np.random.shuffle(data)

for epoch in range(10):
    "random landmarks"
    landmarks = data[int(epoch * 1000):int(epoch * 1000 + 1000), :]
    landmarks[:, 2:] = preprocessing.scale(landmarks[:, 2:])
