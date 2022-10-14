import matplotlib
from utils import *
from tqdm import trange
import probscale
from sklearn.neighbors import KNeighborsRegressor as knn


epochs = 200  # simulation times
show = [10, 15]  # config for show
nlag = 50

'creat locations for simulation result'
loc1 = np.tile(np.arange(1, 336, dtype = int), 335)
loc2 = np.repeat(np.arange(1, 336, dtype = int), 335)
loc = np.hstack((loc1.reshape((-1, 1)), loc2.reshape((-1, 1))))

rawdata = pd.read_csv('./data.csv', header = 0)
rawdata = rawdata.values
rawdata[:, 2:27] = StandardScaler().fit_transform(rawdata[:, 2:27])
data = rawdata[:, :27]

order = np.arange(data.shape[0])
np.random.shuffle(order)
data = data[order, :]

landmark_container = np.empty((500, 27, epochs))
landmark_cdf_container = np.empty((500, 25, epochs))
result_container = np.empty((335 * 335, 25, epochs))  # simulation result container
result_cdf_container = np.empty((335 * 335, 25, epochs))
variogram = np.empty((nlag, 327, epochs))


for epoch in trange(epochs):
    "read data"
    landmarks = data[int(epoch*500):int(epoch*500+500), :27]
    landmark_container[:, :, epoch] = landmarks.copy()

    'convert landmarks to CDF'
    landmarks_cdf = convert_to_cdf(np.copy(landmarks[:, 2:]), show_config=show, if_show=0)
    landmark_cdf_container[:, :, epoch] = landmarks_cdf.copy()

    'Generating MFs'
    if epoch <= 5:
        if_show = 1
    else:
        if_show = 0
    mf_raw, mf_cdf = transport(np.copy(landmarks_cdf), if_show=if_show, show_config=show)

    'Sequential gaussian simulation'
    mf_raw = np.hstack((landmarks[:, 0:2], mf_raw))  # add location
    mf_sim = sgs(mf_raw, if_show = if_show)

    'TPS'
    mf_sim_cdf = convert_to_cdf(mf_sim.copy())
    mf_cdf_lgt = lgt(mf_cdf.copy(), typ = 1)
    lm_cdf_lgt = lgt(landmarks_cdf.copy(), typ = 1)
    mf_sim_cdf_lgt = lgt(mf_sim_cdf.copy(), typ = 1)
    tps = ThinPlateSpline()
    tps.fit(mf_cdf_lgt, lm_cdf_lgt)
    result_cdf_lgt = tps.transform(mf_sim_cdf_lgt)
    result_cdf = lgt(result_cdf_lgt.copy(), typ = -1)
    result = de_cdf(landmarks[:, 2:], landmarks_cdf, result_cdf)
    if if_show == 1:
        plt.imshow(result[:, 10].reshape(335, 335), cmap = 'jet', origin = 'lower')
        plt.colorbar()
        plt.title("SMMT result")
        plt.show()
        plt.imshow(np.flipud(rawdata[:, 12].reshape(335, 335)), cmap = 'jet', origin = 'lower',
                   vmax = np.max(result[:, 10]), vmin = np.min(result[:, 10]))
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
gamma = variogram_gam(rawdata, grid = [335, 335], cellsize = 1, nlag = 50)
plot_variogram(gamma, color = "red")
plot_variogram(variogram)
plt.show()

'show pdf'
common_opts = dict(
    plottype='prob',
    probax='y',
    datascale='log',
    datalabel='Fe',
    scatter_kws=dict(c='g', marker='.', linestyle = 'none', markersize =0.5)
)
for e in range(epochs):
    fig = probscale.probplot(result_container[:, 10, e]+5, dist=None, **common_opts)
common_opts = dict(
    plottype='prob',
    probax='y',
    datascale='log',
    datalabel='Fe',
    scatter_kws=dict(c='r', marker='.', linestyle = 'none', markersize =0.5)
)
fig = probscale.probplot(data[:, 12]+5, dist=None, **common_opts)
plt.show()

'Check result'
e_type = np.mean(result_container, axis = 2).reshape((335, 335, 25))
plt.imshow(e_type[:, :, 10], cmap = 'jet', origin = 'lower')
plt.title("E-type")
plt.colorbar()
plt.show()
std_map = np.std(result_container, axis = 2).reshape((335, 335, 25))
plt.imshow(std_map[:, :, 10], cmap = 'jet', origin = 'lower')
plt.title("STD")
plt.colorbar()
plt.show()


print('result_mean =', np.mean(result_container[:, 10, :]))
print('data_mean =', np.mean(data[:, 12]))
print('result_std =', np.std(result_container[:, 10, :]))
print('data_std =', np.std(data[:, 12]))
pass
