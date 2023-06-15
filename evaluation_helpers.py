import math
import numpy as np
from scipy.stats import norm


def mse(real, pred):
    return np.square(real - pred).mean(axis=1)


def mae(real, pred):
    return np.abs(real - pred).mean(axis=1)


def qlike(real, pred):
    return (np.log(np.square(pred)) + (np.square(real) * np.square(pred))).mean(axis=1)


def mde(real, pred):
    return np.where(np.sign(np.diff(real)) == np.sign(np.diff(pred)), 1, 0).mean(axis=1)


def log_likelihood(real, pred, std):
    return np.log(norm.pdf(pred - real, scale=std)).sum(axis=1)


def log_particle_likelihood(real, pred_particles, std):
    return np.log(norm.pdf(pred_particles - np.expand_dims(real, axis=1), scale=std)).sum()
