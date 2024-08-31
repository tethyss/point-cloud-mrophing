from utils import *

if __name__ == '__main__':
    k = 18
    landmarks = np.load('./result/lm_test.npy')
    landmarks_cdf = convert_to_cdf(landmarks.copy(), show_config=[2, 3], if_show=True, color='b')
    rawdata = np.load('./result/rawdata_test.npy')
    mf_raw_container = np.load('./result/mf_test.npy')
    sim_result = np.load('./result/sgsim_test.npy')
    result_variogram = np.empty((100, 5, 50))
    result_container = sim_result.copy()
    for r in trange(sim_result.shape[2]):
        if_show = False
        if r <= 2:
            if_show = True
        result = TPS(sim_result[:, :, r].copy(), mf_raw_container[:, :, r].copy(), landmarks.copy(), landmarks_cdf.copy(),
                     rawdata.copy(), knn=k, if_show=if_show, show=[2, 3])
        result_container[:, :, r] = result.copy()
        result_variogram[:, :, r] = variogram_gam(result, lag=2, nlag=100, trace=False)
    np.save('./result/vario_SMMT_test.npy', result_variogram)
    np.save('./result/result_SMMT_test.npy', result_container)