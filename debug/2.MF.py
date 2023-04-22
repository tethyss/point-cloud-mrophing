from utils import *

rawdata = np.load('./result/rawdata.npy')
show_config = [10, 12]
label = ['Ag', 'Al', 'Au', 'B', 'Ba', 'Be', 'Bi', 'Ca', 'Co', 'F', 'Fe', 'K',
         'La', 'Li', 'Mg', 'Mn', 'Mo', 'Nb', 'P', 'Sn', 'Sr', 'Ti', 'V', 'Y1', 'Zr']
n_lm = int(335*335*0.1)
lm_loc = np.random.randint(0, 335*335, size=n_lm)
mf_repeat = 50
nlag = 50

"random landmarks"
landmarks = rawdata[lm_loc, :]

'Cumulative distribution of landmarks'
landmarks_cdf = convert_to_cdf(landmarks.copy(), show_config=show_config, if_show=True, color='b')

'Generating MFs'
mf_raw_container = np.zeros((landmarks.shape[0], landmarks.shape[1], mf_repeat))
mf_cdf_container = mf_raw_container.copy()
mf_variogram_container = np.zeros((nlag + 2, sum(range(1, rawdata.shape[1] - 1)) + 2, mf_repeat))
if_show = True
for r in tqdm(range(mf_repeat), position=0, leave=True):
    mf_raw, mf_cdf = transport(landmarks_cdf.copy(), if_show=if_show, show_config=show_config)
    mf_raw_container[:, :, r] = mf_raw.copy()
    mf_cdf_container[:, :, r] = mf_cdf.copy()
    if_show = False
mf_ave = np.sum(mf_raw_container, axis=2) / mf_repeat
plt.scatter(landmarks_cdf[0, show_config[0]], landmarks_cdf[0, show_config[1]], marker='+', c='b', label='landmark point')
plt.scatter(mf_cdf_container[0, show_config[0], :], mf_cdf_container[0, show_config[1], :], marker='x', c='r',
            label='morphing factors')
plt.legend()
plt.title('Pairing of landmark point and morphing factors')
plt.axis('square')
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.show()
