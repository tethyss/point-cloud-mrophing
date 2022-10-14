import matplotlib
import numpy as np

from utils import *
from tqdm import trange
import probscale
from sklearn.neighbors import KNeighborsRegressor as knn

epochs = 200  # simulation times
show = [10, 15]  # config for show
nlm = 561  # number of landmarks
nlag = 50

'creat locations for simulation result'
loc = np.hstack((np.tile(np.arange(1, 336), 335).reshape((-1, 1)), np.repeat(np.arange(1, 336), 335).reshape((-1, 1))))

'read origin data'
rawdata = pd.read_csv('./data.csv', header=0)
rawdata = rawdata.values
rawdata[:, 2:27] = StandardScaler().fit_transform(rawdata[:, 2:27])
data = rawdata[:, :27]
'shuffle data'
order = np.arange(data.shape[0])
np.random.shuffle(order)
data = data[order, :]
data_cdf = convert_to_cdf(data[:, 2:].copy(), show_config=show, if_show=0)

'creat containers'
landmark_container = np.empty((nlm, 27, epochs))  # loc+value
landmark_cdf_container = np.empty((nlm, 25, epochs))
sim_container = np.empty((nlm, 27, epochs))  # loc+value
sim_cdf_container = np.empty((nlm, 25, epochs))
mf_variogram = np.empty((nlag, 327, epochs))
sim_variogram = np.empty((nlag, 327, epochs))
result_container = np.empty((335 * 335, 25, epochs))  # simulation result container
result_cdf_container = np.empty((335 * 335, 25, epochs))
variogram = np.empty((nlag, 327, epochs))

for epoch in trange(epochs):
    if_show = 0
    if epoch <= 6:
        if_show = 1

    "random landmarks"
    landmarks = data[int(epoch * nlm):int(epoch * nlm + nlm), :27]
    landmark_container[:, :, epoch] = landmarks.copy()

    'Cumulative distribution of landmarks'
    landmarks_cdf = data_cdf[int(epoch * nlm):int(epoch * nlm + nlm), :27]
    landmark_cdf_container[:, :, epoch] = landmarks_cdf.copy()

    'Generating MFs'
    mf_raw, mf_cdf = transport(landmarks_cdf.copy(), if_show=if_show, show_config=show)

    'Sequential gaussian simulation'
    mf_raw = np.hstack((landmarks[:, 0:2], mf_raw))  # add location
    mf_sim = sgs(mf_raw, if_show=if_show)

    'calculate variogram of SGSim'
    mf_variogram[:, :, epoch] = variogram_calculation(mf_raw, lag=1, steps=nlag, tol=1, channels=25)
    sim_variogram[:, :, epoch] = variogram_gam(np.hstack((loc, mf_sim)), grid=[335, 335], cellsize=1, nlag=nlag)
    plot_variogram(mf_variogram[:, :, epoch], color="red")
    plot_variogram(sim_variogram[:, :, epoch])
    plt.show()

    'TPS'
    mf_sim_cdf = convert_to_cdf(mf_sim.copy())
    mf_cdf_lgt = lgt(mf_cdf.copy(), typ=1)
    lm_cdf_lgt = lgt(landmarks_cdf.copy(), typ=1)
    mf_sim_cdf_lgt = lgt(mf_sim_cdf.copy(), typ=1)
    tps = ThinPlateSpline()
    tps.fit(mf_cdf_lgt, lm_cdf_lgt)
    result_cdf_lgt = tps.transform(mf_sim_cdf_lgt)
    result_cdf = lgt(result_cdf_lgt.copy(), typ=-1)
    result = de_cdf(landmarks[:, 2:], landmarks_cdf, result_cdf)
    if if_show == 1:
        plt.imshow(result[:, 10].reshape(335, 335), cmap='jet', origin='lower')
        plt.colorbar()
        plt.title("SMMT result")
        plt.show()
        plt.imshow(np.flipud(rawdata[:, 12].reshape(335, 335)), cmap='jet', origin='lower',
                   vmax=np.max(result[:, 10]), vmin=np.min(result[:, 10]))
        plt.colorbar()
        plt.title("Original data")
        plt.show()
    result_container[:, :, epoch] = result.copy()
    result_cdf_container[:, :, epoch] = result_cdf.copy()

    print('computing variogram')
    result = np.hstack((loc, result.copy()))
    result_variogram = variogram_gam(result, grid=[335, 335], cellsize=1, nlag=nlag)
    variogram[:, :, epoch] = result_variogram.copy()

'calculate exhausted variogram'
gamma = variogram_gam(rawdata, grid=[335, 335], cellsize=1, nlag=nlag)
plot_variogram(gamma, color="red")
plot_variogram(variogram)
plt.show()

'show pdf'
common_opts = dict(
    plottype='prob',
    probax='y',
    datascale='log',
    datalabel='Fe',
    scatter_kws=dict(c='g', marker='.', linestyle='none', markersize=0.5)
)
for e in range(epochs):
    fig = probscale.probplot(result_container[:, 10, e] + 5, dist=None, **common_opts)
common_opts = dict(
    plottype='prob',
    probax='y',
    datascale='log',
    datalabel='Fe',
    scatter_kws=dict(c='r', marker='.', linestyle='none', markersize=0.5)
)
fig = probscale.probplot(data[:, 12] + 5, dist=None, **common_opts)
plt.show()

'Check result'
e_type = np.mean(result_container, axis=2).reshape((335, 335, 25))
plt.imshow(e_type[:, :, 10], cmap='jet', origin='lower')
plt.title("E-type")
plt.colorbar()
plt.show()
std_map = np.std(result_container, axis=2).reshape((335, 335, 25))
plt.imshow(std_map[:, :, 10], cmap='jet', origin='lower')
plt.title("STD")
plt.colorbar()
plt.show()

print('result_mean =', np.mean(result_container[:, 10, :]))
print('data_mean =', np.mean(data[:, 12]))
print('result_std =', np.std(result_container[:, 10, :]))
print('data_std =', np.std(data[:, 12]))
pass
