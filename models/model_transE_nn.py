import math

import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda")


class TransE_nn(nn.Module):
    def __init__(self, n_ent, n_rel, depth, margin, norm, hidden, loss):
        super(TransE_nn, self).__init__()
        self.margin = margin
        self.norm = norm
        self.loss = self.get_loss_function(loss)
        self.ent_embedding = nn.Embedding(n_ent, depth)
        self.rel_embedding = nn.Embedding(n_rel, depth)
        self.ent_embedding.weight.data.uniform_(-6 / math.sqrt(depth), 6 / math.sqrt(depth))
        self.rel_embedding.weight.data.uniform_(-6 / math.sqrt(depth), 6 / math.sqrt(depth))
        # self.ent_embedding.weight.data = F.normalize(self.ent_embedding.weight.data, dim=1)
        self.rel_embedding.weight.data = F.normalize(self.rel_embedding.weight.data, dim=1)

        self.hidden_layer = nn.Sequential(
            torch.nn.Linear(2 * depth, hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden, depth),
        )

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
        self.ent_embedding.weight.data = F.normalize(self.ent_embedding.weight.data, dim=1)
        # self.rel_embedding.weight.data = F.normalize(self.rel_embedding.weight.data, dim=1)
        # shape: (batch_size,)
        pos_heads, pos_tails, pos_rels = pos_x[:, 0], pos_x[:, 1], pos_x[:, 2]
        neg_heads, neg_tails, neg_rels = neg_x[:, 0], neg_x[:, 1], neg_x[:, 2]
        pos_score = self.get_score(pos_heads, pos_tails, pos_rels)
        neg_score = self.get_score(neg_heads, neg_tails, neg_rels)
        return torch.max((self.margin + pos_score - neg_score), torch.tensor([0.]).to(device)).mean()

    def get_loss_function(self, mode="margin"):
        if mode == "margin":
            def margin_loss(pos_score, neg_score):
                return torch.max((self.margin + pos_score - neg_score), torch.tensor([0.]).to(device)).mean()
            return margin_loss

