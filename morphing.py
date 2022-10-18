from utils import *
from tqdm import trange
import probscale
from sklearn.neighbors import NearestNeighbors

if_test = False  # 0-full data 1-two dim data
if if_test:
    channel = 2
    show = [2, 3]  # config for show
else:
    channel = 25
    show = [12, 17]

epochs = 100  # simulation times
nlm = 1122  # number of landmarks
nlag = 50  # number of lags in variogram

'creat locations for simulation result'
loc = np.hstack((np.tile(np.arange(1, 336), 335).reshape((-1, 1)), np.repeat(np.arange(1, 336), 335).reshape((-1, 1))))

'read origin data'
rawdata = pd.read_csv('./data.csv', header = 0)
rawdata = rawdata.values
rawdata[:, 2:27] = preprocessing.scale(rawdata[:, 2:27])
if if_test:
    data = rawdata[:, [0, 1, 12, 17]]
else:
    data = rawdata[:, :(2 + channel)]

'variogram of dataset'
exhausted_variogram = variogram_gam(data, [335, 335], 1, 50)
variogram_model = variogram_config(exhausted_variogram.copy())
plot_variogram(exhausted_variogram, name = 'variogram of rawdata by gam', color = 'k', vmodel = variogram_model)
np.savetxt("./variogram of data/variogram of rawdata by gam.csv", exhausted_variogram)

'shuffle data'
order = np.arange(data.shape[0])
np.random.shuffle(order)
data = data[order, :]
data_cdf = convert_to_cdf(data.copy(), show_config = show, if_show = 0)

'creat containers'
landmark_container = np.empty((nlm, 2 + channel, epochs))  # loc+value
landmark_cdf_container = np.empty((nlm, 2 + channel, epochs))
sim_container = np.empty((nlm, 2 + channel, epochs))  # loc+value
sim_cdf_container = np.empty((nlm, 2 + channel, epochs))
mf_variogram = np.empty((nlag, sum(range(1, channel + 1)) + 2, epochs))
sim_variogram = np.empty((nlag, sum(range(1, channel + 1)) + 2, epochs))
result_container = np.empty((335 * 335, 2 + channel, epochs))  # simulation result container
result_cdf_container = np.empty((335 * 335, 2 + channel, epochs))
variogram = np.empty((nlag, sum(range(1, channel + 1)) + 2, epochs))

for epoch in trange(epochs):
    if_show = False
    if epoch <= 6:
        if_show = True

    "random landmarks"
    landmarks = data[int(epoch * nlm):int(epoch * nlm + nlm), :]
    landmark_container[:, :, epoch] = landmarks.copy()

    'variogram of landmarks'
    lm_variogram = variogram_gamv(landmarks, cellsize = 1, nlag = nlag, azm = 0, atol = 180, bandh = 25)
    if epoch < 6:
        plot_variogram(lm_variogram, name = 'variogram of lm epoch ' + str(epoch))

    'Cumulative distribution of landmarks'
    landmarks_cdf = data_cdf[int(epoch * nlm):int(epoch * nlm + nlm), :]
    landmark_cdf_container[:, :, epoch] = landmarks_cdf.copy()

    'Generating MFs'
    mf_raw, mf_cdf = transport(landmarks_cdf.copy(), if_show = False, show_config = show)

    'variogram of morphing factors'
    mf_variogram = variogram_gamv(mf_raw, cellsize = 1, nlag = nlag, azm = 0, atol = 180, bandh = 25)
    plot_variogram(mf_variogram, name = 'variogram of mf epoch ' + str(epoch), color = 'r')

    'Sequential gaussian simulation'
    mf_sim = sgs(mf_raw.copy(), if_show = if_show, vmodel = variogram_model)

    'calculate variogram of SGSim'
    sim_variogram[:, :, epoch] = variogram_gam(mf_sim, grid = [335, 335], cellsize = 1, nlag = nlag)
    if epoch < 6:
        plot_variogram(sim_variogram[:, :, epoch], name = 'variogram of sim result epoch ' + str(epoch))

    'TPS'
    mf_sim_cdf = convert_to_cdf(mf_sim.copy(), show_config = show, if_show = False)
    mf_cdf_lgt = lgt(mf_cdf.copy(), typ = 1)
    lm_cdf_lgt = lgt(landmarks_cdf.copy(), typ = 1)
    mf_sim_cdf_lgt = lgt(mf_sim_cdf.copy(), typ = 1)
    result_cdf_lgt = mf_sim_cdf_lgt.copy()
    nbrs = NearestNeighbors(n_neighbors = 30, algorithm = 'auto').fit(lm_cdf_lgt[:, :2])
    for idx, x in enumerate(mf_sim_cdf_lgt[:, :2]):
        if x % 1000 == 0:
            print('transforming:' + str(x))
        tps = ThinPlateSpline()
        _, indices = nbrs.kneighbors([x])
        tps.fit(mf_cdf_lgt[indices, 2:].reshape(30, -1), lm_cdf_lgt[indices, 2:].reshape(30, -1))
        result_cdf_lgt[idx, 2:] = tps.transform(mf_sim_cdf_lgt[idx, 2:].reshape(1, -1))
    result_cdf = lgt(result_cdf_lgt.copy(), typ = -1)
    result = de_cdf(landmarks[:, 2:], landmarks_cdf[:, 2:], result_cdf[:, 2:])
    if if_show == 1:
        plt.imshow(result[:, 10].reshape(335, 335), cmap = 'jet', origin = 'lower')
        plt.colorbar()
        plt.title("SMMT result")
        plt.show()
        plt.imshow(np.flipud(data[:, 12].reshape(335, 335)), cmap = 'jet', origin = 'lower',
                   vmax = np.max(result[:, 10]), vmin = np.min(result[:, 10]))
        plt.colorbar()
        plt.title("Original data")
        plt.show()
    result_container[:, :, epoch] = result.copy()
    result_cdf_container[:, :, epoch] = result_cdf.copy()

    print('computing variogram')
    variogram[:, :, epoch] = variogram_gam(result, grid = [335, 335], cellsize = 1, nlag = nlag)

'calculate result variogram'
plot_variogram(variogram)

'show pdf'
common_opts = dict(
    plottype = 'prob',
    probax = 'y',
    datascale = 'log',
    datalabel = 'Fe',
    scatter_kws = dict(c = 'g', marker = '.', linestyle = 'none', markersize = 0.5)
)
for e in range(epochs):
    fig = probscale.probplot(result_container[:, 10, e] + 5, dist = None, **common_opts)
common_opts = dict(
    plottype = 'prob',
    probax = 'y',
    datascale = 'log',
    datalabel = 'Fe',
    scatter_kws = dict(c = 'r', marker = '.', linestyle = 'none', markersize = 0.5)
)
fig = probscale.probplot(data[:, 12] + 5, dist = None, **common_opts)
plt.show()

'Check result'
e_type = np.mean(result_container[:, 2:], axis = 2).reshape((335, 335, 25))
plt.imshow(e_type[:, :, 10], cmap = 'jet', origin = 'lower')
plt.title("E-type")
plt.colorbar()
plt.show()
std_map = np.std(result_container[:, 2:], axis = 2).reshape((335, 335, 25))
plt.imshow(std_map[:, :, 10], cmap = 'jet', origin = 'lower')
plt.title("STD")
plt.colorbar()
plt.show()

print('result_mean =', np.mean(result_container[:, 12, :]))
print('data_mean =', np.mean(data[:, 12]))
print('result_std =', np.std(result_container[:, 12, :]))
print('data_std =', np.std(data[:, 12]))
pass
