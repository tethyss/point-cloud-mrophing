from utils import *
from scipy.optimize import curve_fit


def exponential_two(h, r1, r2, c):
    result = np.zeros(h.shape)
    for idx, r in enumerate(h):
        if r <= r1:
            result[idx] = c * (1 - math.exp(-3*r/r1))
        elif r1 <= r <= r2:
            result[idx] = c + (1 - c) * (1 - math.exp(-3*(r-r1) / r2))
        else:
            result[idx] = 1
    return result


def vmodel(variogram, guess=[10, 110, 0.85]):
    x = variogram[1:, 0]
    parameters = np.zeros(variogram.shape[1], 3)
    for i in range(2, variogram.shape[1]):
        y = variogram[1:, i]
        parameters[i-2], _ = curve_fit(exponential_two, x, y, p0=guess)
    return parameters


pass