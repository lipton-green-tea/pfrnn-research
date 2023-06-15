from particle_filter import ParticleFilter

import math
import collections
import numpy as np
from scipy import stats
import torch


# we create a version of the particle filter that can be used
# in the same way as our PFRNNs (i.e. a similar interface)
class ParticleFilterModel:
    def __init__(self, model_config):
        # function generators
        # these will generate any of the functions for a fixed set of parameters
        def particle_initializer_generator(params):
            def particle_initializer(num_particles):
                return [np.random.normal(loc=0, scale=4*params["tau"]) for i in range(0, num_particles)]
            return particle_initializer

        def obvs_prob_func(observation, particle):
            return stats.norm.pdf(observation, loc=0, scale=math.exp(0.5*particle))

        def transformation_func_generator(params):
            def trans_func(particle, control):
                x_mean = params["mu"] + params["phi"] * (particle - params["mu"])
                x = np.random.normal(loc=x_mean, scale=params["tau"])
                return x
            return trans_func

        # pass in our parameters to generate our functions
        trans_func = transformation_func_generator(model_config["parameters"])
        particle_initializer = particle_initializer_generator(model_config["parameters"])

        self.num_particles = model_config["num_particles"]
        self.hidden_dim = 1

        self.pf = ParticleFilter(trans_func, obvs_prob_func, particle_initializer, self.num_particles)


    def forward(self, xs):
        batch_size, seq_len, input_dim = xs.shape

        particles = np.zeros((batch_size, seq_len, self.num_particles, self.hidden_dim))
        estimated_vol = np.zeros((batch_size, seq_len, 1))

        for batch_i in range(batch_size):
            print(batch_i)
            for seq_i in range(seq_len):
                self.pf.step(xs[batch_i, seq_i, 0], None)
                particles[batch_i, seq_i] = np.array(self.pf.particles).reshape((self.num_particles, 1))
                estimated_vol[batch_i, seq_i] = np.array([self.pf.get_mean_particle()])
            self.pf.reset()
        estimated_vol = torch.from_numpy(estimated_vol.transpose((1,0,2)))
        particles = particles.transpose(1, 2, 0, 3).reshape((seq_len, batch_size * self.num_particles, self.hidden_dim))
        particles = torch.from_numpy(particles)
        return estimated_vol, particles
