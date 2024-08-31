from utils import *

rawdata = np.load('./result/bmdata.npy')
show_config = [2, 3]
mf_repeat = 200
nlag = 15

"random landmarks"
cdpoints = np.load('./bmdata/cdpoints.npy')

'Cumulative distribution of landmarks'
cdpoints_cdf = convert_to_cdf(cdpoints, show_config=show_config, if_show=True, color='b')

'Generating MFs'
mf_raw_container = np.zeros((cdpoints.shape[0], cdpoints.shape[1], mf_repeat))
for r in tqdm(range(mf_repeat), position=0, leave=True):
    mf_raw, mf_cdf = transport(cdpoints_cdf, maxit=1e5, if_show=True if r == 0 else False, show_config=show_config)
    mf_raw_container[:, :, r] = mf_raw.copy()

"Calculating variogram of MFs"
n_variogram = int((1 + (mf_raw_container.shape[1] - 2)) * (mf_raw_container.shape[1] - 2) / 2)
M_Dist = ot.dist(mf_raw_container[:, 0:2, 0], mf_raw_container[:, 0:2, 0], metric="euclidean")
variogram_mf = np.zeros((int(nlag + 1), int(n_variogram + 2), mf_repeat))
variogram_mf_avg = np.zeros((int(nlag + 1), int(n_variogram + 2)))
for r in tqdm(range(mf_repeat), position=0, leave=True):
    variogram_mf[:, :, r] += variogram_omni(mf_raw_container[:, :, r], M_Dist, Lag=4, Nlag=nlag, LagTol=2, NumVar=6)
    variogram_mf_avg += variogram_mf[:, :, r]
variogram_mf_avg = variogram_mf_avg / mf_repeat
model = vmodel(variogram_mf_avg[1:, :], guess=[10, 35, 0.6])
plt.rcParams.update({'font.size': 12})
fig, _a = plt.subplots(2, 3, figsize=(8, 4))
plt.subplots_adjust(left=0.05, bottom=0.05, right=0.98, top=0.95, wspace=0.2, hspace=0.2)
axes = _a.flatten()
x = np.linspace(0, np.max(variogram_mf_avg[:, 0]), 400)
ind = [2, 8, 13, 17, 20, 22, 23]
for i in range(6):
    axes[i].plot(variogram_mf_avg[:, 0], variogram_mf[:, ind[i], :], c='orange', alpha=0.2, label='Morphing Factors')
    axes[i].plot(variogram_mf_avg[:, 0], variogram_mf_avg[:, ind[i]], '--', c='r', label='Average Variogram')
    axes[i].plot(x, spherical_two(x, model[i, 0], model[i, 1], model[i, 2]), c='k', label='Variogram Model')
    axes[i].set_title('Variogram of MF' + str(int(i+1)))
    axes[i].set_box_aspect(1 / 2)
    axes[i].set_xlim(0, 60)
    axes[i].set_ylim(0, 1.2)
plt.savefig('./result/variogram_mf.tiff', dpi=330)
plt.show()
np.save('./result/vmodel.npy', model)
np.save('./result/mf.npy', mf_raw_container)
