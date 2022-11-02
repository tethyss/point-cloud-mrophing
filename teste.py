import numpy as np


loc = np.hstack((np.tile(np.arange(1, 336), 335).reshape((-1, 1)), np.repeat(np.arange(1, 336), 335).reshape((-1, 1))))
x=[555,332]
if not max(np.all(loc[:,:2]==x,axis=1)):
    print('not in')
else:
    print('in')
pass