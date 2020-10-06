import math

import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda")


class TransE_nn(nn.Module):
    def __init__(self, n_ent, n_rel, depth, margin, norm, hidden, loss):
        """
        :param n_ent: .type: int
        :param n_rel: .type: int
        :param depth: .type: int
        :param margin: .type: float
        :param norm: .type: [1 | 2]
        :param hidden: .type: list
        :param loss: .type [margin]
        """
        super(TransE_nn, self).__init__()

        self.margin = margin
        self.norm = norm
        self.depth = depth
        self.hidden = hidden
        self.loss_function = self.get_loss_function(loss)
        self.ent_embedding = nn.Embedding(n_ent, depth)
        self.rel_embedding = nn.Embedding(n_rel, depth)
        self.ent_embedding.weight.data.uniform_(-6 / math.sqrt(depth), 6 / math.sqrt(depth))
        self.rel_embedding.weight.data.uniform_(-6 / math.sqrt(depth), 6 / math.sqrt(depth))
        # self.ent_embedding.weight.data = F.normalize(self.ent_embedding.weight.data, dim=1)
        self.rel_embedding.weight.data = F.normalize(self.rel_embedding.weight.data, dim=1)
        self.hidden_layer = self.get_net()

    def get_net(self):
        nets = list()
        nets.append(nn.Linear(2 * self.depth, self.hidden[0]))
        nets.append(nn.ReLU(True))
        for i in range(len(self.hidden) - 1):
            nets.append(nn.Linear(self.hidden[i], self.hidden[i+1]))
            nets.append(nn.ReLU(True))
        nets.append(nn.Linear(self.hidden[-1], self.depth))
        return nn.Sequential(*nets)

    def get_score(self, heads, tails, rels):
        # shape: (batch_size, depth)
        heads, tails, rels = self.ent_embedding(heads), self.ent_embedding(tails), self.rel_embedding(rels)
        # hidden_layer_input: .shape: (batch_size, 2 * depth)
        hidden_layer_input = torch.cat([heads, rels], 1)
        # hidden_layer_output: .shape: (batch_size, depth)
        hidden_layer_output = self.hidden_layer(hidden_layer_input)
        # return shape: (batch_size,)
        return torch.norm(hidden_layer_output - tails, p=self.norm, dim=1)

    def forward(self, pos_x, neg_x):
        # self.ent_embedding.weight.data = F.normalize(self.ent_embedding.weight.data, dim=1)
        # self.rel_embedding.weight.data = F.normalize(self.rel_embedding.weight.data, dim=1)
        # shape: (batch_size,)
        pos_heads, pos_tails, pos_rels = pos_x[:, 0], pos_x[:, 1], pos_x[:, 2]
        neg_heads, neg_tails, neg_rels = neg_x[:, 0], neg_x[:, 1], neg_x[:, 2]
        pos_score = self.get_score(pos_heads, pos_tails, pos_rels)
        neg_score = self.get_score(neg_heads, neg_tails, neg_rels)
        return self.loss_function(pos_score, neg_score)

    def get_loss_function(self, mode="margin"):
        if mode == "margin":
            def margin_loss(pos_score, neg_score):
                return torch.max((self.margin + pos_score - neg_score), torch.tensor([0.]).to(device)).mean()
            return margin_loss
        elif mode == "softplus":
            def softplus_loss(pos_score, neg_score):
                return (torch.log(1 + torch.exp(-1 * pos_score)) + torch.log(1 + torch.exp(neg_score))).mean()
            return softplus_loss

    def get_regularization(self):
        pass
