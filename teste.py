import numpy as np

from utils import *

"""read data
    :arg test:True for tset set
    :return data, show config
"""
rawdata = pd.read_csv('./data.csv', header = 0)
rawdata = rawdata.values
rawdata1 = preprocessing.scale(rawdata[:, 2:27])

rawdata2 = (rawdata[:, 2:27]-np.mean(rawdata[:, 2:27]))/np.std(rawdata[:, 2:27])

pass