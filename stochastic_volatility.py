import math
import numpy as np

class SVL1Paramters():
    def __init__(self, alpha, phi, rho, sigma, initial_innovation=0, initial_volatility=0):
        self.alpha = alpha
        self.phi = phi
        self.rho = rho
        self.sigma = sigma
        self.initial_innovation = initial_innovation
        self.initial_volatility = initial_volatility

class SVL1(object):
    @staticmethod
    def generate_data(timesteps: int, parameters: SVL1Paramters):
        alpha = parameters.alpha
        phi = parameters.phi
        rho = parameters.rho
        sigma = parameters.sigma

        innovations = [parameters.initial_innovation]
        volatility = [parameters.initial_volatility]
        cov_matrix = [
            [1, rho],
            [rho, 1]
        ]
        for x in range(timesteps):
            samples = np.random.multivariate_normal([0,0], cov_matrix)
            ut = samples[0]
            vt1 = samples[1]

            innovations.append(volatility[-1] * ut)
            volatility.append(math.sqrt(math.exp(alpha + phi * math.log(volatility[-1] ** 2) + sigma * vt1)))

        return (volatility, innovations)


# model from Havey, A. (1990). Forecasting, Structural Time Series Models and the Kalman Filter. Cambridge University Press, New York.
class HarveySVParamters():
    def __init__(self, mu, tau, phi):
        self.mu = mu
        self.tau = tau
        self.phi = phi


class HarveySV(object):
    @staticmethod
    def generate_data(timesteps: int, parameters: HarveySVParamters):
        mu = parameters.mu
        tau = parameters.tau
        phi = parameters.phi

        volatility = []
        x0 = np.random.normal(loc=mu, scale=tau)
        volatility.append(x0)
        for x in range(timesteps):
            x_mean = mu + phi * (volatility[-1] - mu)
            x = np.random.normal(loc=x_mean, scale=tau)
            volatility.append(x)

        innovations = [0] + [np.random.normal(loc=0., scale=np.exp(.5 * x)) for x in volatility[1:]]

        return (volatility, innovations)
