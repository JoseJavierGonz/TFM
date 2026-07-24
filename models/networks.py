import torch
import torch.nn as nn
import numpy as np



class Actor_network(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )

        self.mean_layer = nn.Linear(128, action_dim)
        with torch.no_grad():
            self.mean_layer.bias[0] = 1.0
            self.mean_layer.bias[1] = 0.0

        self.std_layer = nn.Linear(128, action_dim)

    def forward(self, state):
        features = self.net(state)

        mean = self.mean_layer(features)
        log_std = self.std_layer(features)
        std = torch.exp(torch.clamp(log_std, -2, 1))
        return mean, std


class Critic_Actor(nn.Module):
    def __init__(self, global_state_dim):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(global_state_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, global_state):
        if global_state.dim() >= 3:
            global_state = global_state.squeeze(1)
        return self.net(global_state)