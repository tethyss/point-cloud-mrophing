from utils import *

'read origin data'
rawdata, show, y_label = read_data(test = 2)

'calculate variogram for rawdata'
rawdata[:, 2:] = preprocessing.scale(rawdata[:, 2:])
rawdata_variogram = variogram_gam(rawdata, cellsize = 4, nlag = 50)

gamma = g2a('gamv_out.out', nlag = 50, num_vario = 3, type = 'gamv')
plot_variogram([rawdata_variogram, gamma], y_label = ['Fe', 'Fe-Mn', 'Mn'], line_label = ['original data', 'selected data'], colors = ['b', 'r'],
               alphas = [1,1], title = 'Variogram of orignial data', vmodel = None)
