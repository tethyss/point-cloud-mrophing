import os
from utils import *

'read origin data'
rawdata, show, y_label = read_data(test=False)

'calculate variogram for rawdata'
if os.path.exists('./result/variogram_exhausted.npy'):
    rawdata_variogram = np.load('./variogram_exhausted.npy')
else:
    pt = preprocessing.PowerTransformer(method = 'box-cox')
    data = rawdata.copy()
    data[:, 2:] = pt.fit_transform(rawdata[:, 2:])
    rawdata_variogram = variogram_gam(data, lag=4, nlag=50, trace=True)
    np.save('./result/variogram_exhausted.npy', rawdata_variogram)
model = vmodel(rawdata_variogram, guess=[0, 10, 110, 0.85])
plot_variogram([rawdata_variogram], ele=len(y_label), y_label=y_label, line_label=['Exhausted data'], colors=['b'],
               alphas=[1], title='variogram of exhausted data', vmodel=None)
np.save('./result/rawdata.npy', rawdata)
