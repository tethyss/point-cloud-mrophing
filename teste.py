import geostatspy.geostats as geostats
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

rawdata = pd.read_excel('./data.xlsx', header=0)
rawdata1=rawdata.copy()
# scaler = StandardScaler()
# rawdata1[rawdata.columns[2:]] = pd.DataFrame(scaler.fit_transform(rawdata[rawdata.columns[2:]]), columns=rawdata.columns[2:])
# rawdata['NAg'], _, _ = geostats.nscore(rawdata, 'Ag')



tmin=-1.0e21
tmax=1.0e21
lag=4
lag_tol=2
nlag=50
azi=0
azi_tol=90
bandwidth=1000
lags, gammas, npps = geostats.gamv(rawdata1,"locx","locy","Ag",tmin,tmax,lag,lag_tol,nlag,azi,azi_tol,bandwidth,isill=1.0)

scatter = plt.scatter(lags,gammas,color = 'darkorange',edgecolor='black',s = 0.05,label = 'Azimuth ' +str(azi))
plt.plot([0,200],[1.0,1.0],color = 'black')
plt.xlabel(r'Lag Distance $\bf(h)$, (m)')
plt.ylabel(r'$\gamma \bf(h)$')
plt.show()
pass