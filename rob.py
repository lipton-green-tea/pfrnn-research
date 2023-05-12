import torch.nn as nn
import torch
from pfrnns import PFLSTMCell, PFGRUCell
import numpy as np
from arguments import parse_args
import collections

class ModelArgs():
    def __init__(self, bpdecay=0.1, l1_weight=0.0, l2_weight=1.0, elbo_weight=0.1, resamp_alpha=0.3):
        self.bpdecay = bpdecay
        self.l1_weight = l1_weight
        self.l2_weight = l2_weight
        self.elbo_weight = elbo_weight
        self.resamp_alpha = resamp_alpha



class SVMParamterEstimator(nn.Module):
    def __init__(self, model_args=dict()):
        super(SVMParamterEstimator, self).__init__()

        self.num_particles = model_args.get("num_particles", 64)
        self.output_dim = 1
        total_emb = model_args.get("input_size", 10)  # should match the size of the input (i.e. observation dimensions)
        self.hidden_dim = model_args.get("hidden_dimension", 100)
        resamp_alpha = 0.1
        self.initialize = 'rand'
        self.model = 'PFLSTM'
        self.dropout_rate = 0.0  # TODO: revert this to have some dropout

        self.rnn = PFLSTMCell(self.num_particles, total_emb,
                    self.hidden_dim, total_emb, total_emb, resamp_alpha)

        self.hnn_dropout = nn.Dropout(self.dropout_rate)

        # TODO: there is a better place to put these variables
        # probably allow the user the pass in a list of numbers
        linear_hidden_size_1 = 150
        linear_hidden_size_2 = 50

        self.layer1 = nn.Sequential(
            nn.Linear(self.hidden_dim, self.output_dim),
            nn.LeakyReLU()
        )

        # self.layer2 = nn.Sequential(
        #     nn.Linear(linear_hidden_size_1, linear_hidden_size_2),
        #     nn.LeakyReLU()
        # )

        # self.layer3 = nn.Sequential(
        #     nn.Linear(linear_hidden_size_2, self.output_dim),
        #     nn.LeakyReLU()
        # )

    def init_hidden(self, batch_size):
        initializer = torch.rand if self.initialize == 'rand' else torch.zeros

        h0 = initializer(batch_size * self.num_particles, self.hidden_dim)  # hidden state
        c0 = initializer(batch_size * self.num_particles, self.hidden_dim)  # cell state
        p0 = torch.ones(batch_size * self.num_particles, 1) * np.log(1 / self.num_particles)  # weights

        hidden = (h0, c0, p0)

        def cudify_hidden(h):
            if isinstance(h, tuple):
                return tuple([cudify_hidden(h_) for h_ in h])
            else:
                return h.cuda()

        if torch.cuda.is_available():
            hidden = cudify_hidden(hidden)

        return hidden

    def forward(self, observations):
        batch_size = observations.size(0)
        embedding = observations

        # create a copy of the input for each particle
        embedding = embedding.repeat(self.num_particles, 1, 1)
        seq_len = embedding.size(1)
        hidden = self.init_hidden(batch_size)

        hidden_states = []
        probs = []


        for step in range(seq_len):
            hidden = self.rnn(embedding[:, step, :], hidden)
            hidden_states.append(hidden[0])
            probs.append(hidden[-1])

            # if step % self.bp_length == 0:
            #     hidden = self.detach_hidden(hidden)
        
        hidden_states = torch.stack(hidden_states, dim=0)
        hidden_states = self.hnn_dropout(hidden_states)

        probs = torch.stack(probs, dim=0)
        prob_reshape = probs.view([seq_len, self.num_particles, -1, 1])
        out_reshape = hidden_states.view([seq_len, self.num_particles, -1, self.hidden_dim])
        y = out_reshape * torch.exp(prob_reshape)
        y = torch.sum(y, dim=1)
        y = self.layer1(y)
        pf_labels = self.layer1(hidden_states)

        # y = self.layer2(y)
        # pf_labels = self.layer2(pf_labels)

        # y = self.layer3(y)
        # pf_labels = self.layer3(pf_labels)

        y_out = y
        pf_out = pf_labels
        
        return y_out, pf_out

    def step(self, obs_in, gt_pos, args):

        pred, particle_pred = self.forward(obs_in)


        # gt_xy_normalized = gt_pos[:, :, :2] / self.map_size
        # gt_theta_normalized = gt_pos[:, :, 2:] / (np.pi * 2)
        # gt_normalized = torch.cat([gt_xy_normalized, gt_theta_normalized], dim=2)
        # for now I will assume that volatility is already normalized
        gt_normalized = gt_pos

        # here we add decay so that earlier predictions in the sequence have less impact on the overall loss
        batch_size = pred.size(1)
        sl = pred.size(0)
        bpdecay_params = np.exp(args.bpdecay * np.arange(sl))
        bpdecay_params = bpdecay_params / np.sum(bpdecay_params)
        if torch.cuda.is_available():
            bpdecay_params = torch.FloatTensor(bpdecay_params).cuda()
        else:
            bpdecay_params = torch.FloatTensor(bpdecay_params)

        bpdecay_params = bpdecay_params.unsqueeze(0)
        bpdecay_params = bpdecay_params.unsqueeze(2)
        pred = pred.transpose(0, 1).contiguous()

        l2_pred_loss = torch.nn.functional.mse_loss(pred, gt_normalized, reduction='none') * bpdecay_params
        l1_pred_loss = torch.nn.functional.l1_loss(pred, gt_normalized, reduction='none') * bpdecay_params

        # separate loss for position and bearing calculation so we can weight their 
        # contribution to the total loss separately
        # l2_xy_loss = torch.sum(l2_pred_loss[:, :, :2])
        # l2_h_loss = torch.sum(l2_pred_loss[:, :, 2])
        # l2_loss = l2_xy_loss + args.h_weight * l2_h_loss
        # we have a 1 element prediction so the above isnt necessary
        l2_loss = torch.mean(l2_pred_loss)

        # l1_xy_loss = torch.mean(l1_pred_loss[:, :, :2])
        # l1_h_loss = torch.mean(l1_pred_loss[:, :, 2])
        # l1_loss = 10*l1_xy_loss + args.h_weight * l1_h_loss
        # we have a 1 element prediction so the above isnt necessary
        l1_loss = torch.mean(l1_pred_loss)

        # separately weight the L1 and L2 loss
        pred_loss = args.l2_weight * l2_loss + args.l1_weight * l1_loss


        total_loss = pred_loss

        # We calculate particle loss using l1 and mse against the actual set of states
        particle_pred = particle_pred.transpose(0, 1).contiguous()
        particle_gt = gt_normalized.repeat(self.num_particles, 1, 1)
        l2_particle_loss = torch.nn.functional.mse_loss(particle_pred, particle_gt, reduction='none') * bpdecay_params
        l1_particle_loss = torch.nn.functional.l1_loss(particle_pred, particle_gt, reduction='none') * bpdecay_params

        # p(y_t| \tau_{1:t}, x_{1:t}, \theta) is assumed to be a Gaussian with variance = 1.
        # other more complicated distributions could be used to improve the performance
        y_prob_l2 = torch.exp(-l2_particle_loss).view(self.num_particles, -1, sl, self.output_dim)
        l2_particle_loss = - y_prob_l2.mean(dim=0).log()

        y_prob_l1 = torch.exp(-l1_particle_loss).view(self.num_particles, -1, sl, self.output_dim)
        l1_particle_loss = - y_prob_l1.mean(dim=0).log()

        # below is not needed as 1 element calculation
        # xy_l2_particle_loss = torch.mean(l2_particle_loss[:, :, :2])
        # h_l2_particle_loss = torch.mean(l2_particle_loss[:, :, 2])
        # l2_particle_loss = xy_l2_particle_loss + args.h_weight * h_l2_particle_loss
        l2_particle_loss = torch.mean(l2_particle_loss)

        # xy_l1_particle_loss = torch.mean(l1_particle_loss[:, :, :2])
        # h_l1_particle_loss = torch.mean(l1_particle_loss[:, :, 2])
        # l1_particle_loss = 10 * xy_l1_particle_loss + args.h_weight * h_l1_particle_loss
        l1_particle_loss = torch.mean(l1_particle_loss)

        belief_loss = args.l2_weight * l2_particle_loss + args.l1_weight * l1_particle_loss
        total_loss = total_loss + args.elbo_weight * belief_loss

        loss_last = torch.nn.functional.mse_loss(pred[:, -1, :], gt_pos[:, -1, :])

        particle_pred = particle_pred.view(self.num_particles, batch_size, sl, self.output_dim)

        return total_loss, loss_last, particle_pred

def generate_xs(parameters, n):
    xs = []
    x0 = np.random.normal(loc=parameters.mu, scale=parameters.tau)
    xs.append(x0)
    for x in range(n):
        x_mean = parameters.mu + parameters.phi * (xs[-1] - parameters.mu)
        x = np.random.normal(loc=x_mean, scale=parameters.tau)
        xs.append(x)
    return np.array(xs)


def generate_ys(xs):
    return np.array([np.random.normal(loc=0., scale=np.exp(.5 * x)) for x in xs[1:]])


n = 5000

Parameters = collections.namedtuple('Parameters', ['mu', 'phi', 'tau'])
parameters = Parameters(mu=2. * np.log(.7204), phi=.9807, tau=1.1489)

xs = generate_xs(parameters, n)

ys = generate_ys(xs)

# normalize values to between 0 and 1
xs = (xs - np.min(xs)) / (np.max(xs) - np.min(xs))
ys = (ys - np.min(ys)) / (np.max(ys) - np.min(ys))

# below we create several windows of observations
window_size = 10
windows = ( # add arrays together to create sub_windows
    np.expand_dims(np.arange(window_size), 0) +  # time offsets
    np.expand_dims(np.arange(len(ys) - window_size), 0).T  # start times
)
obvs = ys[windows]

# get rid of the volatilies for which we do not have enough previous values for
xs = xs[window_size+1:]

if __name__=="__main__":
    model = SVMParamterEstimator()
    optimizer = torch.optim.RMSprop(
            model.parameters(), lr=0.005)
    # set model to train mode
    model.train()
    obs = torch.from_numpy(np.array([obvs]).astype(np.float32))
    obs.type(torch.FloatTensor)
    actual = torch.tensor([[[x] for x in xs.astype(np.float32)]])
    actual.type(torch.FloatTensor)
    args = parse_args()
    model.zero_grad()
    print(obs.dtype)
    print(actual.dtype)
    print(obs.shape)
    for x in range(0,10):
        model.zero_grad()  # set all gradients to zero before each iteration
        loss, log_loss, particle_pred = model.step(obs,actual,args)
        print(loss)
        loss.backward()
        optimizer.step()

    # set model to eval mode before evaluating
    model.eval()
    pred, particle_pred = model.forward(obs)
