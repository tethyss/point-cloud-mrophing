from utils import *

if __name__ == '__main__':
    k = 18
    nlag=15
    cdpoints = np.load('./bmdata/cdpoints.npy')
    cdpoints_cdf = convert_to_cdf(cdpoints.copy(), show_config=[2, 3], if_show=False, color='b')
    bmdata = np.load('./result/bmdata.npy')
    mf = np.load('./result/mf.npy')
    sim_result = np.load('./result/sgsim_test.npy')
    result_variogram = np.empty((nlag+1, mf.shape[1], mf.shape[2]))
    result_container = sim_result.copy()
    for r in trange(sim_result.shape[2]):
        result = TPS(sim_result[:, :, r].copy(), mf[:, :, r].copy(), cdpoints.copy(), cdpoints_cdf.copy(),
                     bmdata.copy(), knn=k, if_show=False if r > 2 else True, show=[2, 3])
        result_container[:, :, r] = result.copy()
        result_variogram[:, :, r] = variogram_gam(result, lag=2, nlag=100, trace=False)
    np.save('./result/vario_SMMT_test.npy', result_variogram)
    np.save('./result/result_SMMT_test.npy', result_container)