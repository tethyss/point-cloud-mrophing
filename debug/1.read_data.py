import os
from utils import *

'read origin data'
rawdata, show, y_label = read_data(test=False)

'calculate variogram for rawdata'
if os.path.exists('./result/variogram_exhausted.npy'):
    rawdata_variogram = np.load('./result/variogram_exhausted.npy')
else:
    rawdata[:, 2:] = preprocessing.scale(rawdata[:, 2:])
    rawdata_variogram = variogram_gam(rawdata, 4, 50, trace=False)
    np.save('./result/variogram_exhausted.npy', rawdata_variogram)
plot_variogram([rawdata_variogram], ele=len(y_label), y_label=y_label, line_label=['Exhausted data'], colors=['b'],
               alphas=[1], title='variogram of exhausted data', vmodel=None)
if not os.path.exists('./result/rawdata.npy'):
    np.save('./result/rawdata.npy', rawdata)
