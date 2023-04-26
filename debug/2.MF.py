from utils import *

rawdata = np.load('./result/rawdata.npy')
show_config = [10, 12]
mf_repeat = 50
nlag = 50
y_label = ['Ag', 'Al', 'Au', 'B', 'Ba', 'Be', 'Bi', 'Ca', 'Co', 'F', 'Fe', 'K',
           'La', 'Li', 'Mg', 'Mn', 'Mo', 'Nb', 'P', 'Sn', 'Sr', 'Ti', 'V', 'Y1', 'Zr']


for landmark_portion in [0.1, 0.2]:
    lm_loc = np.random.randint(0, 335 * 335, size = int(335 * 335 * landmark_portion))

    "random landmarks"
    landmarks = rawdata[lm_loc, :]
    lm_scale = landmarks.copy()
    lm_scale[:, 2:] = preprocessing.scale(landmarks[:, 2:])
    variogram_lm = variogram_gamv(lm_scale, cellsize = 4, nlag = 50, azm = 0, atol = 22.5, dbglevel = 0)

    'Cumulative distribution of landmarks'
    landmarks_cdf = convert_to_cdf(landmarks.copy(), show_config = show_config, if_show = True, color = 'b')

    'Generating MFs'
    mf_raw_container = np.zeros((landmarks.shape[0], landmarks.shape[1], mf_repeat))
    mf_cdf_container = mf_raw_container.copy()
    mf_variogram_container = np.zeros((nlag + 2, sum(range(1, rawdata.shape[1] - 1)) + 2, mf_repeat))
    for r in tqdm(range(mf_repeat), position = 0, leave = True):
        if_show = False
        if r < 3:
            if_show = True
        mf_raw, mf_cdf = transport(landmarks_cdf.copy(), if_show = if_show, show_config = show_config)
        mf_raw_container[:, :, r] = mf_raw.copy()
        mf_cdf_container[:, :, r] = mf_cdf.copy()
        mf_variogram_container[:, :, r] = variogram_gamv(mf_raw, cellsize = 4, nlag = 50, azm = 0, atol = 22.5,
                                                         dbglevel = 0)
    show_connect(landmarks_cdf, mf_cdf_container, show_config=[10, 12])
    variogram_ave = np.sum(mf_variogram_container, axis = 2) / mf_repeat
    model = vmodel(variogram_ave, guess = [0, 10, 110, 0.85])
    plot_variogram([variogram_lm, mf_variogram_container, variogram_ave], ele = len(y_label), y_label = y_label,
                   line_label = ['landmark', 'mf', 'mf_ave'], colors = ['b', 'orange', 'k'],
                   alphas = [1, 0.2, 1], title = 'variogram of morphing factors '+str(landmark_portion), vmodel = model)
    np.save('./result/lm_' + str(landmark_portion) + '.npy', landmarks)
    np.save('./result/mf_' + str(landmark_portion) + '.npy', mf_raw_container)
    np.save('./result/vm_' + str(landmark_portion) + '.npy', model)