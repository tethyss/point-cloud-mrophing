import numpy as np
import pandas as pd

from utils import *


def a2g(data):
    print("generating GSLIB file")
    columns = ['X', 'Y', 'Ag', 'Al', 'Au', 'B', 'Ba', 'Be', 'Bi', 'Ca', 'Co', 'F', 'Fe', 'K', 'La', 'Li', 'Mg',
               'Mn', 'Mo', 'Nb', 'P', 'Sn', 'Sr', 'Ti', 'V', 'Y1', 'Zr']
    df=[]
    df.append(np.asarray(columns).reshape((-1, 1)))
    df = pd.DataFrame(df)
    df.to_csv('1test.dat', index = False, header = None)
    pass





data, landmarks = read_data(plot=0)
a2g(data)
pass
