from utils import *
from tqdm import trange, tqdm
import probscale
from sklearn.neighbors import NearestNeighbors

epochs = 100  # simulation times
nlm = 1122  # number of landmarks
lag = 4
nlag = 50  # number of lags in variogram
mf_repeat = 100

'creat locations for simulation result'
loc = np.hstack((np.tile(np.arange(1, 336), 335).reshape((-1, 1)), np.repeat(np.arange(1, 336), 335).reshape((-1, 1))))

'read origin data'
data, show = read_data(test = True)

'shuffle data'
np.random.shuffle(data)

'creat containers'
sim_container = np.empty((nlm, data.shape[1], epochs))  # loc+value
sim_cdf_container = np.empty((nlm, data.shape[1], epochs))
mf_variogram = np.empty((nlag, sum(range(1, data.shape[1] - 1)) + 2, epochs))
sim_variogram = np.empty((nlag, sum(range(1, data.shape[1] - 1)) + 2, epochs))
result_container = np.empty((335 * 335, data.shape[1], epochs))  # simulation result container
result_cdf_container = np.empty((335 * 335, data.shape[1], epochs))
variogram = np.empty((nlag, sum(range(1, data.shape[1] - 1)) + 2, epochs))

for epoch in tqdm(range(epochs), position = 0, leave = False):
    if_show = False
    if epoch <= 6:
        if_show = True

    "random landmarks"
    landmarks = data[int(epoch * nlm):int(epoch * nlm + nlm), :]
    landmarks[:, 2:] = preprocessing.scale(landmarks[:, 2:])

    'variogram of landmarks'
    lm_variogram = variogram_gamv(landmarks, cellsize = lag, nlag = nlag, azm = 0, atol = 180, dbglevel = 0)
    if epoch < 6:
        plot_variogram(lm_variogram, names = ['Fe', 'Fe-Mn', 'Mn'],
                       name = 'variogram of landmarks-epoch ' + str(epoch), color = 'r')

    'Cumulative distribution of landmarks'
    landmarks_cdf = convert_to_cdf(landmarks.copy(), show_config = show, if_show = False, color = 'b')

    'Generating MFs'
    mf_raw_container = np.zeros((landmarks.shape[0], landmarks.shape[1], mf_repeat))
    mf_variogram_container = np.zeros((nlag + 2, sum(range(1, data.shape[1] - 1)) + 2, mf_repeat))
    for r in tqdm(range(mf_repeat), position = 0):
        mf_raw, _ = transport(landmarks_cdf.copy(), if_show = False, show_config = show)
        mf_variogram = variogram_gamv(mf_raw, cellsize = lag, nlag = nlag, azm = 0, atol = 180, dbglevel = 0)
        mf_raw_container[:, :, r] = mf_raw
        mf_variogram_container[:, :, r] = mf_variogram
    mf_ave = np.sum(mf_raw_container, axis = 2) / mf_repeat
    variogram_ave = np.sum(mf_variogram_container, axis = 2) / mf_repeat
    mf_variogram = variogram_gamv(mf_ave, cellsize = lag, nlag = nlag, azm = 0, atol = 180, dbglevel = 0)
    vmodel = variogram_config(variogram_ave)
    v_models = create_vmodel(vmodel, lag, nlag)
    plot_variogram1([mf_variogram_container, variogram_ave, v_models], names = ['MF1', 'MF1-MF2', 'MF2'],
                    name = 'variogram-epoch ' + str(epoch))


    'Sequential gaussian simulation'
    mf_sim = sgs(mf_ave.copy(), if_show = if_show, vmodel = vmodel)

    'calculate variogram of SGSim'
    sim_variogram[:, :, epoch] = variogram_gam(mf_sim, cellsize = lag, nlag = nlag)
    if epoch < 6:
        plot_variogram(sim_variogram[:, :, epoch], name = 'variogram of sim result-epoch ' + str(epoch))

    'TPS'
    mf_sim_cdf = convert_to_cdf(mf_sim.copy(), show_config = show, if_show = False)
    mf_cdf_lgt = lgt(mf_cdf.copy(), typ = 1)
    lm_cdf_lgt = lgt(landmarks_cdf.copy(), typ = 1)
    mf_sim_cdf_lgt = lgt(mf_sim_cdf.copy(), typ = 1)
    result_cdf_lgt = mf_sim_cdf_lgt.copy()
    nbrs = NearestNeighbors(n_neighbors = 90, algorithm = 'auto').fit(lm_cdf_lgt[:, :2])
    for idx, x in enumerate(mf_sim_cdf_lgt[:, :2]):
        tps = ThinPlateSpline()
        _, indices = nbrs.kneighbors([x])
        tps.fit(mf_cdf_lgt[indices, 2:].reshape(30, -1), lm_cdf_lgt[indices, 2:].reshape(30, -1))
        result_cdf_lgt[idx, 2:] = tps.transform(mf_sim_cdf_lgt[idx, 2:].reshape(1, -1))
    result_cdf = lgt(result_cdf_lgt.copy(), typ = -1)
    result = de_cdf(landmarks[:, 2:], landmarks_cdf[:, 2:], result_cdf[:, 2:])
    result = np.hstack((result_cdf[:, :2], result))
    result1 = result[np.lexsort((result[:, 0], result[:, 1])), :].copy()
    result.argsort()
    if if_show:
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
    variogram[:, :, epoch] = variogram_gam(result, cellsize = 1, nlag = nlag)
    plot_variogram(variogram[:, :, epoch], name = 'variogram of result epoch ' + str(epoch))

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
