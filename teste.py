from utils import *
import time


def read_data():
    full_data = pd.read_csv('./data.csv', header = 0)
    full_data = full_data.values  # remove the location
    full_data = full_data[np.lexsort((full_data[:, 0], full_data[:, 1])), :]
    full_data = np.reshape(full_data, [335, 335, 25])
    deposits = pd.read_csv('/content/drive/MyDrive/deposit.csv', header = 0)
    deposits = deposits.values
    return full_data, deposits


if __name__ == '__main__':
    data, deposit = read_data()
pass
