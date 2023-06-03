import math
import numpy as np
from scipy.stats import norm


def mse(real, pred):
    return np.square(real - pred).mean()


def mae(real, pred):
    return np.abs(real - pred).mean()


def qlike(real, pred):
    return (np.log(np.square(pred)) + (np.square(real) * np.square(pred))).mean()


def mde(real, pred):
    return np.where(np.sign(np.diff(real)) == np.sign(np.diff(pred)), 1, 0).mean()


def log_likelihood(real, pred, std):
    return np.log(norm.pdf(pred - real, scale=std)).sum()


def log_particle_likelihood(real, pred_particles, std):
    return np.log(norm.pdf(pred_particles - np.expand_dims(real, axis=1), scale=std)).sum()
