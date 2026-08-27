import torch
import torch.nn as nn
import numpy as np



class Actor_network(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )

        self.camera_encoder = nn.Sequential(
            nn.Linear(6,16),
            nn.ReLU(),
            nn.Linear(16,16),
            nn.ReLU()
        )

        self.fusion = nn.Sequential(
            nn.Linear(80,128),
            nn.ReLU(),
            nn.Linear(128,128),
            nn.ReLU()
        )

        self.mean_layer = nn.Linear(128, action_dim)
        # with torch.no_grad():
        #     self.mean_layer.bias[0] = 1.0
        #     self.mean_layer.bias[1] = 0.0

        self.std_layer = nn.Linear(128, action_dim)

    def forward(self, state, cam_state):
        features = self.net(state)
        camera = self.camera_encoder(cam_state)
        fusion = torch.cat([features, camera], dim=-1)
        fusion = self.fusion(fusion)

        mean = self.mean_layer(fusion)
        log_std = self.std_layer(fusion)
        #std = torch.exp(torch.clamp(log_std, -2, 1))
        std = torch.exp(torch.clamp(log_std, -4, -0.5))
        return mean, std


class Critic_Actor(nn.Module):
    def __init__(self, global_state_dim):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(global_state_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )

        self.camera_encoder = nn.Sequential(
                    nn.Linear(12,32),
                    nn.ReLU(),      
                    nn.Linear(32,32),
                    nn.ReLU()
                )
        
        self.fusion = nn.Sequential(
            nn.Linear(160, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, global_state, cam_state):
        if global_state.dim() >= 3:
            global_state = global_state.squeeze(1)
        if cam_state.dim() == 3:
            cam_state = cam_state.squeeze(1)

        features = self.net(global_state)
        camera = self.camera_encoder(cam_state)
        fusion = torch.cat([features, camera], dim=-1)
        fusion = self.fusion(fusion)
        
        return fusion