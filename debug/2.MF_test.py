from utils import *

rawdata = np.load('./result/rawdata_test.npy')
show_config = [2, 3]
landmark_portion = 0.1
lm_loc = list(range(int(335*335)))
np.random.shuffle(lm_loc)
mf_repeat = 50
nlag = 50

"random landmarks"
landmarks = rawdata[lm_loc[:int(335*335*landmark_portion)], :]

'Cumulative distribution of landmarks'
landmarks_cdf = convert_to_cdf(landmarks.copy(), show_config=show_config, if_show=True, color='b')

'Generating MFs'
mf_raw_container = np.zeros((landmarks.shape[0], landmarks.shape[1], mf_repeat))
mf_variogram_container = np.zeros((nlag + 2, sum(range(1, rawdata.shape[1] - 1)) + 2, mf_repeat))
if_show = True
for r in tqdm(range(mf_repeat), position=0, leave=True):
    mf_raw, mf_cdf = transport(landmarks_cdf.copy(), maxit=1e6, if_show=if_show, show_config=show_config)
    mf_raw_container[:, :, r] = mf_raw.copy()
    if_show = False

np.save('./result/lm_test.npy', landmarks)
np.save('./result/mf_test.npy', mf_raw_container)
