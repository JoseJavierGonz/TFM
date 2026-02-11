import torch
import torch.nn as nn
import numpy as np

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
        #Inicializar bias para tender al movimiento al principio
        with torch.no_grad():
            self.mean_layer.bias[0] = 2.0   
            self.mean_layer.bias[1] = 0.0  
            self.mean_layer.bias[2] = -4.0 

        self.std_layer = nn.Linear(128, action_dim)

    def forward(self, state):
        first_layers = self.net(state)
        mean = self.mean_layer(first_layers)
        throttle = torch.sigmoid(mean[:, 0:1])
        steer = torch.tanh(mean[:, 1:2])
        brake = torch.sigmoid(mean[:, 2:3])
        mean = torch.cat([throttle, steer, brake], dim=1)
        log_std = self.std_layer(first_layers)
        std = torch.sigmoid(log_std) * 0.5 + 0.1  # std ∈ [0.1, 0.5]
        
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