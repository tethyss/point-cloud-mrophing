import numpy as np

data = np.empty((200, 200, 6))

for i in range(6):
    data[:, :, i] = np.loadtxt('./benchmark/Reference_Z' + str(i+1) + '_numpy.txt')

landmark = np.loadtxt('./benchmark/Conditioning_Data_6dim_Numpy.txt')
pass
