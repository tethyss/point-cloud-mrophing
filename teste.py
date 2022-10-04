import scipy.stats as stats
import numpy as np
dist = stats.truncnorm(-4.1, 4.1, loc=0, scale=1)
x = (dist.rvs(200)).reshape((-1, 1))
for e in range(25 - 1):
    x = np.hstack((x, (dist.rvs(200)).reshape((-1, 1))))
    print(np.mean(x[:, e+1]))
    print(np.std(x[:, e+1]))
pass
