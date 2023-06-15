import math
from copy import deepcopy
import numpy as np


# Defining the Particle Filter Class

# This is a partial/generalized definition of a particle filter
# To create the final particle filter the following functions need to be specified
# 1. The transformation func: describes how our particles get from the current state to the next, including using the control. 
# 2. The observation probability func: this describes how likely our particle is to exist given the current observation
# 3. The particle initializer: takes in a number of particles to create and creates those new particles, which therefore also determines the shape/size of the particles

# There are also two other considerations to make (which are encoded into the three functions above).
# 1. the size/shape of the hidden state
# 2. what we provide as input

class ParticleFilter():
    def __init__(self, transformation_func, observation_probability, particle_initializer, num_particles):
        self.transformation_func = transformation_func
        self.observation_probability = observation_probability
        self.num_particles = num_particles
        self.particle_initializer = particle_initializer
        self.particles = self.particle_initializer(num_particles)
        self.weights = [(1/num_particles)] * num_particles

    
    def reset(self):
        self.particles = self.particle_initializer(self.num_particles)
        self.weights = [(1/self.num_particles)] * self.num_particles


    def step(self, observation, control):
        self.update_particles(observation, control)
        self.resample()


    def update_particles(self, observation, control):
        min_weight = 1 / (self.num_particles ** 2)
        self.particles = [self.transformation_func(self.particles[i], control) for i in range(self.num_particles)]
        self.weights = [math.sqrt(self.weights[i] * self.observation_probability(observation, self.particles[i])) for i in range(self.num_particles)]
        self.weights = [max(w, min_weight) for w in self.weights]
        self.weights = [(w / sum(self.weights)) for w in self.weights]


    def resample(self):
        indices = np.random.choice(np.arange(0, self.num_particles), size=self.num_particles, p=self.weights)
        new_particles = []
        for i in indices:
            new_particles.append(deepcopy(self.particles[i]))
        self.particles = new_particles
        self.weights = [(1/self.num_particles)] * self.num_particles

    def get_mean_particle(self):
        return sum(self.particles) / len(self.particles)
