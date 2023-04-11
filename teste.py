import numpy as np

from utils import *

'read origin data'
rawdata = np.empty((200, 200, 6))
for i in range(6):
    rawdata[:, :, i] = np.loadtxt('./benchmark/Reference_Z' + str(i + 1) + '_numpy.txt')
'add location'
data = np.empty((200 * 200, 8))
for i in range(200 * 200):
    data[i, :] = np.hstack((int(i % 200), int(i // 200), rawdata[i // 200, i % 200, :]))
# data[:,2:]=(data[:,2:]-np.mean(data[:,2:], axis=0))/np.std(data[:,2:], axis=0)

"random landmarks"
landmarks = np.loadtxt('./benchmark/Conditioning_Data_6dim_Numpy.txt')
# landmarks[:, 2:] = (landmarks[:, 2:] - np.mean(landmarks[:, 2:], axis=0)) / np.std(landmarks[:, 2:], axis=0)

landmarks_cdf = convert_to_cdf(landmarks.copy(), show_config=[2, 3], if_show=True, color='b')

mf_raw_container = np.load('mf_raw.npy')
mf_sim = np.load('result1.npy')

result_container = np.empty((40000, 8, 100))
for r in tqdm(range(100)):
    show = False
    if r <= 1:
        show = True
    result = TPS(mf_sim[:, :, r], mf_raw_container[:, :, r].copy(), landmarks, landmarks_cdf, data
                 , knn=18, if_show=show, show=[2, 3], add=False)
    result_container[:, :, r] = result.copy()

result_etype = np.mean(result_container, axis=2)
plt.imshow(result_etype[:, 2].reshape(200, 200), cmap='jet', origin='lower')
plt.show()
result_sd = np.std(result_container, axis=2)
plt.imshow(result_sd[:, 2].reshape(200, 200), cmap='jet', origin='lower')
plt.scatter(landmarks[:, 0], landmarks[:, 1], c='none', edgecolor='grey')
plt.show()
