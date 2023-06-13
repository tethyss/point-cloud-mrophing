import os
from utils import *


def plot_variogram(variograms):
    fig, axs = plt.subplots(1, 3, figsize = (15, 2.5))
    plt.subplots_adjust(left = 0.08, bottom = 0.05, right = 0.98, top = 0.95, wspace = 0.5, hspace = 0.2)
    axs[0].plot(variograms[:, 0], variograms[:, 2], linewidth = 1.5, color = 'b')
    axs[0].tick_params(axis = 'both', labelsize = 15)
    axs[0].set_box_aspect(1 / 2)
    axs[0].set_ylim(0, 1.5)
    axs[1].plot(variograms[:, 0], variograms[:, 3], linewidth = 1.5, color = 'b')
    axs[1].tick_params(axis = 'both', labelsize = 15)
    axs[1].set_box_aspect(1 / 2)
    axs[1].set_ylim(-0.75, 0.75)
    axs[2].plot(variograms[:, 0], variograms[:, 4], linewidth = 1.5, color = 'b')
    axs[2].tick_params(axis = 'both', labelsize = 15)
    axs[2].set_box_aspect(1 / 2)
    axs[2].set_ylim(0, 1.5)
    plt.savefig('./variogram of data/variogram of test rawdata.pdf', dpi = 330)
    plt.show()


'read origin data'
rawdata, show, y_label = read_data(test=True)

'calculate variogram for rawdata'
if os.path.exists('./result/variogram_exhausted_test.npy'):
    rawdata_variogram = np.load('./result/variogram_exhausted_test.npy')
else:
    scaler = preprocessing.StandardScaler()
    data = rawdata.copy()
    data[:, 2:] = scaler.fit_transform(rawdata[:, 2:])
    rawdata_variogram = variogram_gam(data, lag=2, nlag=100, trace=True)
    np.save('./result/variogram_exhausted_test.npy', rawdata_variogram)
model = vmodel(rawdata_variogram, guess=[0, 10, 110, 0.85])
plot_variogram(rawdata_variogram)
np.save('./result/rawdata_test.npy', rawdata)
