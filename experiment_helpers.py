import numpy as np


def mse(real, pred):
    return np.square(real - pred).mean()


def qlike(real, pred):
    return (np.log(pred**2) + np.square(real) * np.square(pred)).mean()