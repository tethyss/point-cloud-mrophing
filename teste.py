from utils import *
import time


def TPS(sim, mf, lm, lm_cdf, rawdata, knn, if_show, show):
    mf_cdf = convert_to_cdf(mf, show_config=show, if_show=False)
    mf_sim_cdf = convert_to_cdf(sim, show_config=show, if_show=False)
    mf_cdf_lgt = lgt(mf_cdf.copy(), typ=1)
    lm_cdf_lgt = lgt(lm_cdf, typ=1)
    mf_sim_cdf_lgt = lgt(mf_sim_cdf.copy(), typ=1)
    mf_base = mf_cdf_lgt.copy()
    lm_base = lm_cdf_lgt.copy()
    np.random.shuffle(mf_sim_cdf_lgt)
    result_cdf_lgt = mf_sim_cdf_lgt.copy()
    for idx, x in enumerate(tqdm(mf_sim_cdf_lgt[:, :2], position=0, leave=False)):
        if not max(np.all(lm_base[:, :2] == x, axis=1)):
            loc = search_box(x, lm_base, knn)
            nbrs = NearestNeighbors(n_neighbors=knn, algorithm='auto').fit(lm_base[loc, :2])
            tps = ThinPlateSpline()
            *_, indices = nbrs.kneighbors([x])
            tps.fit(mf_base[loc[indices], 2:].reshape(knn, -1), lm_base[loc[indices], 2:].reshape(knn, -1))
            result_cdf_lgt[idx, 2:] = tps.transform(mf_sim_cdf_lgt[idx, 2:].reshape(1, -1))
            mf_base = np.vstack((mf_base, mf_sim_cdf_lgt[idx]))
            lm_base = np.vstack((lm_base, result_cdf_lgt[idx]))
    result_cdf = lgt(result_cdf_lgt.copy(), typ=-1)
    result = de_cdf(lm[:, 2:], lm_cdf[:, 2:], result_cdf[:, 2:])
    result = np.hstack((result_cdf[:, :2], result))
    result = result[np.lexsort((result[:, 0], result[:, 1])), :].copy()
    if if_show:
        plt.imshow(result[:, 2].reshape(335, 335), cmap='jet', origin='lower')
        plt.colorbar()
        plt.title("SMMT result")
        plt.show()
        plt.imshow(rawdata[:, 2].reshape(335, 335), cmap='jet', origin='lower',
                   vmax=np.max(result[:, 2]), vmin=np.min(result[:, 2]))
        plt.colorbar()
        plt.title("Original data")
        plt.show()
    return result, result_cdf


def search_box(x, pool, knn):
    density = (335*335)/len(pool)
    loc=[]
    rate = 1
    while len(loc) <= knn:
        r = math.ceil(math.sqrt(knn*density)*rate)
        x0 = max(0, int(x[0]-r))
        x1 = min(335, int(x[0]+r))
        y0 = max(0, int(x[1] - r))
        y1 = min(335, int(x[1] + r))
        loc = np.where((x0 <= pool[:, 0]) & (pool[:, 0] <= x1) & (y0 <= pool[:, 1]) & (pool[:, 1] <= y1))
        loc = np.asarray(loc).reshape(-1)
        rate = rate*1.05
    return loc


if __name__ == '__main__':
    mf_raw = np.load('./transdata/mf_raw.npy')
    mf_sim = np.load('./transdata/mf_sim.npy')
    landmarks = np.load('./transdata/landmarks.npy')
    rawdata = np.load('./transdata/rawdata.npy')
    landmarks_cdf = np.load('./transdata/landmarks_cdf.npy')
    t1=time.time()
    result, result_cdf = TPS(mf_sim, mf_raw, landmarks, landmarks_cdf, rawdata
                             , knn=20, if_show=True, show=[2, 3])
    print(time.time()-t1)
pass
