from utils import *

mf_raw_container = np.load('./result/mf_test.npy')
rawdata_variogram = np.load('./result/variogram_exhausted_test.npy')
model = vmodel(rawdata_variogram, guess=[0, 10, 110, 0.85])
mf_repeat = 50
result_container = np.empty((335 * 335, 4, mf_repeat))  # simulation result container
result_variogram = np.empty((50, 5, mf_repeat))

'Sequential gaussian simulation'
for r in tqdm(range(mf_repeat), position=0, leave=True):
    if_show = False
    if r <= 2:
        if_show = True
    mf_sim = sgs(mf_raw_container[:, :, r].copy(), if_show=if_show, vmodel=model)
    result_container[:, :, r] = mf_sim.copy()
    result_variogram[:, :, r] = variogram_gam(mf_sim, lag=4, nlag=50, trace=False)