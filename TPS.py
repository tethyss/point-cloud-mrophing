import cv2
from utils import *


def thin_plate_transform(x, y, offw, offh, imshape, shift_l=-0.05, shift_r=0.05, num_points=5, offsetMatrix=False):
    rand_p = np.random.choice(x.size, num_points, replace = False)
    movingPoints = np.zeros((1, num_points, 2), dtype = 'float32')
    fixedPoints = np.zeros((1, num_points, 2), dtype = 'float32')

    movingPoints[:, :, 0] = x[rand_p]
    movingPoints[:, :, 1] = y[rand_p]
    fixedPoints[:, :, 0] = movingPoints[:, :, 0] + offw * (np.random.rand(num_points) * (shift_r - shift_l) + shift_l)
    fixedPoints[:, :, 1] = movingPoints[:, :, 1] + offh * (np.random.rand(num_points) * (shift_r - shift_l) + shift_l)

    tps = cv2.createThinPlateSplineShapeTransformer()
    good_matches = [cv2.DMatch(i, i, 0) for i in range(num_points)]
    tps.estimateTransformation(movingPoints, fixedPoints, good_matches)

    imh, imw = imshape
    x, y = np.meshgrid(np.arange(imw), np.arange(imh))
    x, y = x.astype('float32'), y.astype('float32')
    newxy = tps.applyTransformation(np.dstack((x.ravel(), y.ravel())))[1]
    newxy = newxy.reshape([imh, imw, 2])

    if offsetMatrix:
        return newxy, newxy - np.dstack((x, y))
    else:
        return newxy
