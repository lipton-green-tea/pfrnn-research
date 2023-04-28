import torch.nn as nn
import torch
from rob_pfrnns import PFRNN
import numpy as np


class HarveySVPF(nn.Module):
    def __init__(self, model_args=dict()):
        super(HarveySVPF, self).__init__()

        self.num_particles = model_args.get("num_particles", 64)
        self.output_dim = 1
        total_emb = 1
        self.hidden_dim = 1
        resamp_alpha = 0.3
        self.initialize = 'rand'
        self.model = 'PFLSTM'
        self.dropout_rate = 0.0  # TODO: revert this to have some dropout
        self.use_gpu = model_args.get("use_gpu", True)

        self.rnn = PFRNN(self.num_particles, total_emb,
                    self.hidden_dim, total_emb, resamp_alpha, use_gpu=self.use_gpu)

        self.hnn_dropout = nn.Dropout(self.dropout_rate)


    def init_hidden(self, batch_size):
        initializer = torch.rand if self.initialize == 'rand' else torch.zeros

        h0 = initializer(batch_size * self.num_particles, self.hidden_dim)  # hidden state

        # we want to sample to be drawn from a distribution from -2 to 2
        h0 = (h0 * 4) - 2

        p0 = torch.ones(batch_size * self.num_particles, 1) * (1 / self.num_particles)  # weights

        hidden = (h0, p0)

        def cudify_hidden(h):
            if isinstance(h, tuple):
                return tuple([cudify_hidden(h_) for h_ in h])
            else:
                return h.to('cuda')

        if torch.cuda.is_available() and self.use_gpu:
            hidden = cudify_hidden(hidden)

        return hidden


    def detach_hidden(self, hidden):
        if isinstance(hidden, tuple):
            return tuple([h.detach() for h in hidden])
        else:
            return hidden.detach()


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
        y = out_reshape * prob_reshape
        y = torch.sum(y, dim=1)

        y_out = y
        pf_out = hidden_states
        
        return y_out, pf_out

    def step(self, obs_in, true_vol, args):

        pred, particle_pred = self.forward(obs_in)

        # here we add decay so that earlier predictions in the sequence have less impact on the overall loss
        batch_size = pred.size(1)
        sl = pred.size(0)
        bpdecay_params = np.exp(args.bpdecay * np.arange(sl))
        bpdecay_params = bpdecay_params / np.sum(bpdecay_params)
        if torch.cuda.is_available() and self.use_gpu:
            bpdecay_params = torch.FloatTensor(bpdecay_params).to('cuda')
        else:
            bpdecay_params = torch.FloatTensor(bpdecay_params)

        bpdecay_params = bpdecay_params.unsqueeze(0)
        bpdecay_params = bpdecay_params.unsqueeze(2)
        pred = pred.transpose(0, 1).contiguous()

        l2_pred_loss = torch.nn.functional.mse_loss(pred, true_vol, reduction='none') * bpdecay_params
        l1_pred_loss = torch.nn.functional.l1_loss(pred, true_vol, reduction='none') * bpdecay_params

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
        particle_gt = true_vol.repeat(self.num_particles, 1, 1)
        l2_particle_loss = torch.nn.functional.mse_loss(particle_pred, particle_gt, reduction='none') * bpdecay_params
        l1_particle_loss = torch.nn.functional.l1_loss(particle_pred, particle_gt, reduction='none') * bpdecay_params

        # p(y_t| \tau_{1:t}, x_{1:t}, \theta) is assumed to be a Gaussian with variance = 1.
        # other more complicated distributions could be used to improve the performance
        y_prob_l2 = torch.exp(-l2_particle_loss).view(self.num_particles, -1, sl, self.output_dim)
        l2_particle_loss = - y_prob_l2.mean(dim=0).log()

        y_prob_l1 = torch.exp(-l1_particle_loss).view(self.num_particles, -1, sl, self.output_dim)
        l1_particle_loss = - y_prob_l1.mean(dim=0).log()

        l2_particle_loss = torch.mean(l2_particle_loss)

        l1_particle_loss = torch.mean(l1_particle_loss)

        belief_loss = args.l2_weight * l2_particle_loss + args.l1_weight * l1_particle_loss
        total_loss = total_loss + args.elbo_weight * belief_loss

        loss_last = torch.nn.functional.mse_loss(pred[:, -1, :], true_vol[:, -1, :])

        particle_pred = particle_pred.view(self.num_particles, batch_size, sl, self.output_dim)

        return total_loss, loss_last, particle_pred
