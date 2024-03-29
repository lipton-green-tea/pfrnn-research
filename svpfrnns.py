import torch
from torch import nn
import numpy as np

class PFRNN(nn.Module):
    def __init__(self, num_particles, input_size=1, hidden_size=1, ext_obs=1, resamp_alpha=0.1):
        """
        :param num_particles: number of particles for a PF-RNN
        :param input_size: the size of input x_t
        :param hidden_size: the size of the hidden particle h_t^i
        :param ext_obs: the size for o_t(x_t)
        :param ext_act: the size for u_t(x_t)
        :param resamp_alpha: the control parameter \alpha for soft-resampling.
        We use the importance sampling with a proposal distribution q(i) = \alpha w_t^i + (1 - \alpha) (1 / K)
        """
        super(PFRNN, self).__init__()
        self.num_particles = num_particles
        self.input_size = input_size
        self.h_dim = hidden_size
        self.ext_obs = ext_obs
        self.resamp_alpha = resamp_alpha
        self.train_obs_model = True
        self.train_trans_model = True
        self.load_from_pretrained = True

        # self.fc_obs = nn.Linear(self.ext_obs + self.h_dim, 1)
        self.fc_obs_l1 = nn.Linear(self.ext_obs + self.h_dim, 100)
        self.fc_obs_l2 = nn.Linear(100, 100)
        self.fc_obs_l3 = nn.Linear(100, 1)

        # init weights to approximate a normal pdf
        if self.load_from_pretrained:
            fc_obs_l1_weights = np.transpose(np.load("./models/layer_0_weights.npy"))
            fc_obs_l1_biases = np.transpose(np.load("./models/layer_0_biases.npy"))
            fc_obs_l2_weights = np.transpose(np.load("./models/layer_1_weights.npy"))
            fc_obs_l2_biases = np.transpose(np.load("./models/layer_1_biases.npy"))
            fc_obs_l3_weights = np.transpose(np.load("./models/layer_2_weights.npy"))
            fc_obs_l3_biases = np.transpose(np.load("./models/layer_2_biases.npy"))
            self.fc_obs_l1.weight.data = torch.from_numpy(fc_obs_l1_weights)
            self.fc_obs_l1.bias.data = torch.from_numpy(fc_obs_l1_biases)
            self.fc_obs_l2.weight.data = torch.from_numpy(fc_obs_l2_weights)
            self.fc_obs_l2.bias.data = torch.from_numpy(fc_obs_l2_biases)
            self.fc_obs_l3.weight.data = torch.from_numpy(fc_obs_l3_weights)
            self.fc_obs_l3.bias.data = torch.from_numpy(fc_obs_l3_biases)
        self.fc_obs_l1.weight.requires_grad = self.train_obs_model
        self.fc_obs_l1.bias.requires_grad = self.train_obs_model
        self.fc_obs_l2.weight.requires_grad = self.train_obs_model
        self.fc_obs_l2.bias.requires_grad = self.train_obs_model
        self.fc_obs_l3.weight.requires_grad = self.train_obs_model
        self.fc_obs_l3.bias.requires_grad = self.train_obs_model
        self.fc_obs = nn.Sequential(
            self.fc_obs_l1,
            nn.Sigmoid(),
            self.fc_obs_l2,
            nn.Sigmoid(),
            self.fc_obs_l3,
            nn.Sigmoid(),
        )

        self.batch_norm = nn.BatchNorm1d(self.num_particles)

        # init the layers for our transition function
        # this will take as input a hidden state (volatility) and normaly dist. random float
        # TODO: let it take parameter values as input

        self.fc_trans_l1 = nn.Linear(self.h_dim + 1, 100)
        self.fc_trans_l2 = nn.Linear(100, 100)
        self.fc_trans_l3 = nn.Linear(100, 1)

        if self.load_from_pretrained:
            fc_trans_l1_weights = np.transpose(np.load("./models/trans_layer_0_weights.npy"))
            fc_trans_l1_biases = np.transpose(np.load("./models/trans_layer_0_biases.npy"))
            fc_trans_l2_weights = np.transpose(np.load("./models/trans_layer_1_weights.npy"))
            fc_trans_l2_biases = np.transpose(np.load("./models/trans_layer_1_biases.npy"))
            fc_trans_l3_weights = np.transpose(np.load("./models/trans_layer_2_weights.npy"))
            fc_trans_l3_biases = np.transpose(np.load("./models/trans_layer_2_biases.npy"))
            self.fc_trans_l1.weight.data = torch.from_numpy(fc_trans_l1_weights)
            self.fc_trans_l1.bias.data = torch.from_numpy(fc_trans_l1_biases)
            self.fc_trans_l2.weight.data = torch.from_numpy(fc_trans_l2_weights)
            self.fc_trans_l2.bias.data = torch.from_numpy(fc_trans_l2_biases)
            self.fc_trans_l3.weight.data = torch.from_numpy(fc_trans_l3_weights)
            self.fc_trans_l3.bias.data = torch.from_numpy(fc_trans_l3_biases)
        self.fc_obs_l1.weight.requires_grad = self.train_trans_model
        self.fc_obs_l1.bias.requires_grad = self.train_trans_model
        self.fc_obs_l2.weight.requires_grad = self.train_trans_model
        self.fc_obs_l2.bias.requires_grad = self.train_trans_model
        self.fc_obs_l3.weight.requires_grad = self.train_trans_model
        self.fc_obs_l3.bias.requires_grad = self.train_trans_model
        self.fc_trans = nn.Sequential(
            self.fc_trans_l1,
            nn.Sigmoid(),
            self.fc_trans_l2,
            nn.Sigmoid(),
            self.fc_trans_l3
        )

    def resampling(self, particles, prob):
        """
        The implementation of soft-resampling. We implement soft-resampling in a batch-manner.

        :param particles: \{(h_t^i, c_t^i)\}_{i=1}^K for PF-LSTM and \{h_t^i\}_{i=1}^K for PF-GRU.
                        each tensor has a shape: [num_particles * batch_size, h_dim]
        :param prob: weights for particles in the log space. Each tensor has a shape: [num_particles * batch_size, 1]
        :return: resampled particles and weights according to soft-resampling scheme.
        """
        resamp_prob = self.resamp_alpha * prob + (1 - self.resamp_alpha) * 1 / self.num_particles
        resamp_prob = resamp_prob.view(self.num_particles, -1)
        indices = torch.multinomial(resamp_prob.transpose(0, 1),
                                    num_samples=self.num_particles, replacement=True)
        batch_size = indices.size(0)
        indices = indices.transpose(1, 0).contiguous()
        offset = torch.arange(batch_size).type(torch.LongTensor).unsqueeze(0)
        if torch.cuda.is_available():
            offset = offset.cuda()
        indices = offset + indices * batch_size
        flatten_indices = indices.view(-1, 1).squeeze()

        # PFLSTM
        if type(particles) == tuple:
            particles_new = (particles[0][flatten_indices],
                             particles[1][flatten_indices])
        # PFGRU
        else:
            particles_new = particles[flatten_indices]

        prob_new = torch.exp(prob.view(-1, 1)[flatten_indices])
        prob_new = prob_new / (self.resamp_alpha * prob_new + (1 -
                                                               self.resamp_alpha) / self.num_particles)
        prob_new = prob_new.view(self.num_particles, -1, 1)
        prob_new = prob_new / torch.sum(prob_new, dim=0, keepdim=True)
        prob_new = prob_new.view(-1, 1)

        return particles_new, prob_new


    def forward(self, input_, hx):
        h0, p0 = hx
        batch_size = h0.size(0)

        if torch.cuda.is_available():
            random_input = torch.cuda.FloatTensor(h0.shape).normal_()
        else:
            random_input = torch.FloatTensor(h0.shape).normal_()
        h1 = self.fc_trans(torch.concat((h0, random_input), dim=1))
        
        obs_liklihood = self.fc_obs(torch.concat((h1, input_), dim=1))
        p1 = obs_liklihood.view(self.num_particles, -1, 1) * \
            p0.view(self.num_particles, -1, 1)

        p1 = p1 / torch.sum(p1, dim=0, keepdim=True)

        h1, p1 = self.resampling(h1, p1)

        return h1, p1
    

class SmallPFRNN(nn.Module):
    def __init__(self, num_particles, input_size=1, hidden_size=1, ext_obs=1, resamp_alpha=0.1):
        """
        :param num_particles: number of particles for a PF-RNN
        :param input_size: the size of input x_t
        :param hidden_size: the size of the hidden particle h_t^i
        :param ext_obs: the size for o_t(x_t)
        :param ext_act: the size for u_t(x_t)
        :param resamp_alpha: the control parameter \alpha for soft-resampling.
        We use the importance sampling with a proposal distribution q(i) = \alpha w_t^i + (1 - \alpha) (1 / K)
        """
        super(SmallPFRNN, self).__init__()
        self.num_particles = num_particles
        self.input_size = input_size
        self.h_dim = hidden_size
        self.ext_obs = ext_obs
        self.resamp_alpha = resamp_alpha
        self.train_obs_model = True
        self.train_trans_model = True

        # self.fc_obs = nn.Linear(self.ext_obs + self.h_dim, 1)
        self.fc_obs_l1 = nn.Linear(self.ext_obs + self.h_dim, 40)
        self.fc_obs_l2 = nn.Linear(40, 25)
        self.fc_obs_l3 = nn.Linear(25, 1)
        self.fc_obs = nn.Sequential(
            self.fc_obs_l1,
            nn.Sigmoid(),
            self.fc_obs_l2,
            nn.Sigmoid(),
            self.fc_obs_l3,
            nn.Sigmoid(),
        )

        self.batch_norm = nn.BatchNorm1d(self.num_particles)

        # init the layers for our transition function
        # this will take as input a hidden state (volatility) and normaly dist. random float
        # TODO: let it take parameter values as input

        self.fc_trans_l1 = nn.Linear(self.h_dim, 25)
        self.fc_trans_l2 = nn.Linear(25, 15)
        self.fc_trans_l3 = nn.Linear(15, 1)
        self.fc_trans = nn.Sequential(
            self.fc_trans_l1,
            nn.Sigmoid(),
            self.fc_trans_l2,
            nn.Sigmoid(),
            self.fc_trans_l3,
        )

        # here we define a simple layer to determine the variance to use for reparameterization
        self.fc_var_l1 = nn.Linear(self.h_dim + 1, 1)
        self.fc_var = nn.Sequential(
            self.fc_var_l1,
            nn.Sigmoid(),
        )

    def resampling(self, particles, prob):
        """
        The implementation of soft-resampling. We implement soft-resampling in a batch-manner.

        :param particles: \{(h_t^i, c_t^i)\}_{i=1}^K for PF-LSTM and \{h_t^i\}_{i=1}^K for PF-GRU.
                        each tensor has a shape: [num_particles * batch_size, h_dim]
        :param prob: weights for particles in the log space. Each tensor has a shape: [num_particles * batch_size, 1]
        :return: resampled particles and weights according to soft-resampling scheme.
        """
        resamp_prob = self.resamp_alpha * prob + (1 - self.resamp_alpha) * 1 / self.num_particles
        resamp_prob = resamp_prob.view(self.num_particles, -1)
        indices = torch.multinomial(resamp_prob.transpose(0, 1),
                                    num_samples=self.num_particles, replacement=True)
        batch_size = indices.size(0)
        indices = indices.transpose(1, 0).contiguous()
        offset = torch.arange(batch_size).type(torch.LongTensor).unsqueeze(0)
        if torch.cuda.is_available():
            offset = offset.cuda()
        indices = offset + indices * batch_size
        flatten_indices = indices.view(-1, 1).squeeze()

        # PFLSTM
        if type(particles) == tuple:
            particles_new = (particles[0][flatten_indices],
                             particles[1][flatten_indices])
        # PFGRU
        else:
            particles_new = particles[flatten_indices]

        prob_new = torch.exp(prob.view(-1, 1)[flatten_indices])
        prob_new = prob_new / (self.resamp_alpha * prob_new + (1 -
                                                               self.resamp_alpha) / self.num_particles)
        prob_new = prob_new.view(self.num_particles, -1, 1)
        prob_new = prob_new / torch.sum(prob_new, dim=0, keepdim=True)
        prob_new = prob_new.view(-1, 1)

        return particles_new, prob_new


    def forward(self, input_, hx):
        h0, p0 = hx
        batch_size = h0.size(0)

        h1 = self.fc_trans(h0)

        # add noise
        var = self.fc_var(torch.concat((h0, input_), dim=1))
        std = torch.nn.functional.softplus(var)
        if torch.cuda.is_available():
            eps = torch.cuda.FloatTensor(std.shape).normal_() / 3
        else:
            eps = torch.FloatTensor(std.shape).normal_() / 3
        h1 = h1 + std * eps
        
        obs_liklihood = self.fc_obs(torch.concat((h1, input_), dim=1))
        p1 = obs_liklihood.view(self.num_particles, -1, 1) * \
            p0.view(self.num_particles, -1, 1)

        p1 = p1 / torch.sum(p1, dim=0, keepdim=True)

        h1, p1 = self.resampling(h1, p1)

        return h1, p1


class GarchPFRNN(nn.Module):
    def __init__(self, num_particles, input_size=2, hidden_size=5, ext_obs=1, resamp_alpha=0.1):
        """
        :param num_particles: number of particles for a PF-RNN
        :param input_size: the size of input x_t
        :param hidden_size: the size of the hidden particle h_t^i
        :param ext_obs: the size for o_t(x_t)
        :param ext_act: the size for u_t(x_t)
        :param resamp_alpha: the control parameter \alpha for soft-resampling.
        We use the importance sampling with a proposal distribution q(i) = \alpha w_t^i + (1 - \alpha) (1 / K)
        """
        super(GarchPFRNN, self).__init__()
        self.num_particles = num_particles
        self.input_size = input_size
        self.h_dim = hidden_size
        self.ext_obs = ext_obs
        self.resamp_alpha = resamp_alpha
        self.train_obs_model = True
        self.train_trans_model = True
        self.load_from_pretrained = True

        # self.fc_obs = nn.Linear(self.ext_obs + self.h_dim, 1)
        self.fc_obs_l1 = nn.Linear(self.ext_obs + self.h_dim, 100)
        self.fc_obs_l2 = nn.Linear(100, 100)
        self.fc_obs_l3 = nn.Linear(100, 1)

        # init weights to approximate a normal pdf
        if self.load_from_pretrained:
            fc_obs_l1_weights = np.transpose(np.load("./models/layer_0_weights.npy"))
            fc_obs_l1_biases = np.transpose(np.load("./models/layer_0_biases.npy"))
            fc_obs_l2_weights = np.transpose(np.load("./models/layer_1_weights.npy"))
            fc_obs_l2_biases = np.transpose(np.load("./models/layer_1_biases.npy"))
            fc_obs_l3_weights = np.transpose(np.load("./models/layer_2_weights.npy"))
            fc_obs_l3_biases = np.transpose(np.load("./models/layer_2_biases.npy"))
            self.fc_obs_l1.weight.data = torch.from_numpy(fc_obs_l1_weights)
            self.fc_obs_l1.bias.data = torch.from_numpy(fc_obs_l1_biases)
            self.fc_obs_l2.weight.data = torch.from_numpy(fc_obs_l2_weights)
            self.fc_obs_l2.bias.data = torch.from_numpy(fc_obs_l2_biases)
            self.fc_obs_l3.weight.data = torch.from_numpy(fc_obs_l3_weights)
            self.fc_obs_l3.bias.data = torch.from_numpy(fc_obs_l3_biases)
        self.fc_obs_l1.weight.requires_grad = self.train_obs_model
        self.fc_obs_l1.bias.requires_grad = self.train_obs_model
        self.fc_obs_l2.weight.requires_grad = self.train_obs_model
        self.fc_obs_l2.bias.requires_grad = self.train_obs_model
        self.fc_obs_l3.weight.requires_grad = self.train_obs_model
        self.fc_obs_l3.bias.requires_grad = self.train_obs_model
        self.fc_obs = nn.Sequential(
            self.fc_obs_l1,
            nn.Sigmoid(),
            self.fc_obs_l2,
            nn.Sigmoid(),
            self.fc_obs_l3,
            nn.Sigmoid(),
        )

        self.batch_norm = nn.BatchNorm1d(self.num_particles)

        # init the layers for our transition function
        # this will take as input a hidden state (volatility) and normaly dist. random float
        # TODO: let it take parameter values as input

        self.fc_trans_l1 = nn.Linear(self.h_dim, 200)
        self.fc_trans_l2 = nn.Linear(200, 200)
        self.fc_trans_l3 = nn.Linear(200, 200)
        self.fc_trans_l4 = nn.Linear(200, 100)
        self.fc_trans_l5 = nn.Linear(100, 1)

        if self.load_from_pretrained:
            fc_trans_l1_weights = np.transpose(np.load("./pretrained_weights/garch_trans_layer_0_weights.npy"))
            fc_trans_l1_biases = np.transpose(np.load("./pretrained_weights/garch_trans_layer_0_biases.npy"))
            fc_trans_l2_weights = np.transpose(np.load("./pretrained_weights/garch_trans_layer_1_weights.npy"))
            fc_trans_l2_biases = np.transpose(np.load("./pretrained_weights/garch_trans_layer_1_biases.npy"))
            fc_trans_l3_weights = np.transpose(np.load("./pretrained_weights/garch_trans_layer_2_weights.npy"))
            fc_trans_l3_biases = np.transpose(np.load("./pretrained_weights/garch_trans_layer_2_biases.npy"))
            fc_trans_l4_weights = np.transpose(np.load("./pretrained_weights/garch_trans_layer_3_weights.npy"))
            fc_trans_l4_biases = np.transpose(np.load("./pretrained_weights/garch_trans_layer_3_biases.npy"))
            fc_trans_l5_weights = np.transpose(np.load("./pretrained_weights/garch_trans_layer_4_weights.npy"))
            fc_trans_l5_biases = np.transpose(np.load("./pretrained_weights/garch_trans_layer_4_biases.npy"))
            self.fc_trans_l1.weight.data = torch.from_numpy(fc_trans_l1_weights)
            self.fc_trans_l1.bias.data = torch.from_numpy(fc_trans_l1_biases)
            self.fc_trans_l2.weight.data = torch.from_numpy(fc_trans_l2_weights)
            self.fc_trans_l2.bias.data = torch.from_numpy(fc_trans_l2_biases)
            self.fc_trans_l3.weight.data = torch.from_numpy(fc_trans_l3_weights)
            self.fc_trans_l3.bias.data = torch.from_numpy(fc_trans_l3_biases)
            self.fc_trans_l4.weight.data = torch.from_numpy(fc_trans_l4_weights)
            self.fc_trans_l4.bias.data = torch.from_numpy(fc_trans_l4_biases)
            self.fc_trans_l5.weight.data = torch.from_numpy(fc_trans_l5_weights)
            self.fc_trans_l5.bias.data = torch.from_numpy(fc_trans_l5_biases)
        self.fc_trans_l1.weight.requires_grad = self.train_trans_model
        self.fc_trans_l1.bias.requires_grad = self.train_trans_model
        self.fc_trans_l2.weight.requires_grad = self.train_trans_model
        self.fc_trans_l2.bias.requires_grad = self.train_trans_model
        self.fc_trans_l3.weight.requires_grad = self.train_trans_model
        self.fc_trans_l3.bias.requires_grad = self.train_trans_model
        self.fc_trans_l4.weight.requires_grad = self.train_trans_model
        self.fc_trans_l4.bias.requires_grad = self.train_trans_model
        self.fc_trans_l5.weight.requires_grad = self.train_trans_model
        self.fc_trans_l5.bias.requires_grad = self.train_trans_model

        self.fc_trans = nn.Sequential(
            self.fc_trans_l1,
            nn.Tanh(),
            self.fc_trans_l2,
            nn.Tanh(),
            self.fc_trans_l3,
            nn.Tanh(),
            self.fc_trans_l4,
            nn.Tanh(),
            self.fc_trans_l5,
        )

        # here we define a simple layer to determine the variance to use for reparameterization
        self.fc_var_l1 = nn.Linear(self.h_dim + self.input_size, 1)
        self.fc_var_l1.weight.data = torch.zeros_like(self.fc_var_l1.weight)
        self.fc_var_l1.bias.data = torch.ones_like(self.fc_var_l1.bias) * 0.00005
        self.fc_var = nn.Sequential(
            self.fc_var_l1,
            nn.Sigmoid(),
        )

    def resampling(self, particles, prob):
        """
        The implementation of soft-resampling. We implement soft-resampling in a batch-manner.

        :param particles: \{(h_t^i, c_t^i)\}_{i=1}^K for PF-LSTM and \{h_t^i\}_{i=1}^K for PF-GRU.
                        each tensor has a shape: [num_particles * batch_size, h_dim]
        :param prob: weights for particles in the log space. Each tensor has a shape: [num_particles * batch_size, 1]
        :return: resampled particles and weights according to soft-resampling scheme.
        """
        resamp_prob = self.resamp_alpha * prob + (1 - self.resamp_alpha) * 1 / self.num_particles
        resamp_prob = resamp_prob.view(self.num_particles, -1)
        indices = torch.multinomial(resamp_prob.transpose(0, 1),
                                    num_samples=self.num_particles, replacement=True)
        batch_size = indices.size(0)
        indices = indices.transpose(1, 0).contiguous()
        offset = torch.arange(batch_size).type(torch.LongTensor).unsqueeze(0)
        if torch.cuda.is_available():
            offset = offset.cuda()
        indices = offset + indices * batch_size
        flatten_indices = indices.view(-1, 1).squeeze()

        # PFLSTM
        if type(particles) == tuple:
            particles_new = (particles[0][flatten_indices],
                             particles[1][flatten_indices])
        # PFGRU
        else:
            particles_new = particles[flatten_indices]

        prob_new = torch.exp(prob.view(-1, 1)[flatten_indices])
        prob_new = prob_new / (self.resamp_alpha * prob_new + (1 -
                                                               self.resamp_alpha) / self.num_particles)
        prob_new = prob_new.view(self.num_particles, -1, 1)
        prob_new = prob_new / torch.sum(prob_new, dim=0, keepdim=True)
        prob_new = prob_new.view(-1, 1)

        return particles_new, prob_new


    def forward(self, input_, hx):
        h0, p0 = hx
        batch_size = h0.size(0)
        var = torch.Tensor([0.0028861 , 0.07228504, 0.07229742, 0.1444811, 0.01155304, 0.02310782 , 0.02304983])
        mean = torch.Tensor([ 4.97861683e-03 , 1.54884455e-01 , 1.24940350e-01 , 7.50835131e-01 , 1.99980894e-02 , 7.63748711e-05 , -2.50800646e-05])
        trans_input_normed = (torch.concat((h0, input_), dim=1) - mean) / var
        h1_vol = self.fc_trans(trans_input_normed)
        var = torch.Tensor([0.00291117])
        mean = torch.Tensor([0.00552819])
        h1_vol = h1_vol * var + mean
        h1 = torch.concat((h0[:, :-1], h1_vol), dim=1)

        # add noise 
        var = self.fc_var(torch.concat((h0, input_), dim=1))
        std = torch.nn.functional.softplus(var)
        if torch.cuda.is_available():
            eps = torch.cuda.FloatTensor(std.shape).normal_()
        else:
            eps = torch.FloatTensor(std.shape).normal_()
        #h1[:, -1:] = h1[:, -1:] + std * eps  # only add noise to volatility estimation for now
        
        obs_liklihood = self.fc_obs(torch.concat((h1[:,-1:], input_[:,-1:]), dim=1))
        p1 = obs_liklihood.view(self.num_particles, -1, 1) * \
            p0.view(self.num_particles, -1, 1)

        p1 = p1 / torch.sum(p1, dim=0, keepdim=True)

        h1, p1 = self.resampling(h1, p1)

        return h1, p1
