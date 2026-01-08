import torch
import torch.nn as nn

class Actor_network(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )

        self.mean_layer = nn.Linear(128, action_dim)

        self.std_layer = nn.Linear(128, action_dim)

    def forward(self, state):
        first_layers = self.net(state)
        mean = self.mean_layer(first_layers)
        log_std = self.std_layer(first_layers)
        std = torch.exp(log_std)
        return mean, std
    


class Critic_Actor(nn.Module):
    def __init__(self, state_dim):
        super().__init__()  
        self.features = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, state):
        return self.features(state)