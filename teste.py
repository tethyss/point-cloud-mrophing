import matplotlib.pyplot as plt
import numpy as np
import probscale

result_container = np.load("result.npy")
result_cdf_container = np.load('result_cdf.npy')
common_opts = dict(
    plottype='prob',
    probax='y',
    datascale='log',
    datalabel='Fe',
    scatter_kws=dict(c='g', marker='.', linestyle = 'none', markersize =1)
)
for e in range(result_container.shape[2]):
    fig = probscale.probplot(result_container[:, 10, e]+5, dist=None, **common_opts)
    # plt.scatter(result_container[:, 10, e], result_cdf_container[:, 10, e], c='g', s=5)
plt.show()
