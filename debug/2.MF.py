from utils import *

rawdata = np.load('./result/rawdata.npy')
show_config = [10, 12]
mf_repeat = 50
nlag = 50
for landmark_portion in [0.05, 0.1, 0.2, 0.3]:
    lm_loc = np.random.randint(0, 335*335, size=int(335*335*landmark_portion))

    "random landmarks"
    landmarks = rawdata[lm_loc, :]

    'Cumulative distribution of landmarks'
    landmarks_cdf = convert_to_cdf(landmarks.copy(), show_config=show_config, if_show=False, color='b')

    'Generating MFs'
    mf_raw_container = np.zeros((landmarks.shape[0], landmarks.shape[1], mf_repeat))
    mf_cdf_container = mf_raw_container.copy()
    mf_variogram_container = np.zeros((nlag + 2, sum(range(1, rawdata.shape[1] - 1)) + 2, mf_repeat))
    if_show = False
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
