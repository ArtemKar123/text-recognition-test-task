import numpy as np


def box_height(box):
    return abs(box[3][1] - box[0][1])


def box_width(box):
    return abs(box[1][0] - box[0][0])


def box_center(box):
    return np.mean(box, axis=0).astype(float)
