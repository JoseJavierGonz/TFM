import torch
import torch.nn as nn
import numpy as np



class Actor_network(nn.Module):
    """Red neuronal de los actores"""
    def __init__(self, state_dim, action_dim):
        """Constructor de la red.
        Consiste en una cabeza para el estado del vehiculo
        Otra para los vehiculos detectados
        Y una red fusión."""
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
        nn.init.constant_(self.mean_layer.bias, 0.0)
        #aqui es donde vamos a gestionar el bias, lo inicializamos a 0.5
        #buscando que tienda al movimiento. Despues pondremos un decay porque
        #si no, nunca aprende a frenar
        self.register_buffer('action_offset', torch.tensor([0.5, 0.0]))
        #queremos acotar la media porque si no los gradientes rompen el entrenamiento
        #limitamos mas el steering porque si no zigzaguea
        self.register_buffer('mean_scale', torch.tensor([2.5, 1.2]))
  
        self.log_std = nn.Parameter(torch.tensor([-0.7, -1.2]))
        #también queremos garantizar una exploración mínima para no tener una red determinista
        self.register_buffer('log_std_floor', torch.tensor([-1.0, -2.0]), persistent=False)

    def forward(self, state, cam_state):
        """Step por la red"""
        features = self.net(state)
        camera = self.camera_encoder(cam_state)
        fusion = torch.cat([features, camera], dim=-1)
        fusion = self.fusion(fusion)

        raw = self.mean_layer(fusion)
        mean = self.mean_scale * torch.tanh(raw / self.mean_scale) + self.action_offset
        #al igual que hemos garantizado una exploración mínima, queremos cortar para que las acciones 
        #no crezcan llenandose de ruido
        std = torch.exp(torch.maximum(self.log_std, self.log_std_floor).clamp(max=0.0)).expand_as(mean)
        return mean, std
    #decaimiento del aceleardor
    def decay_throttle_bias(self, current_step, decay_steps):
        """Decaimiento del bias en el acelerador"""
        new_bias = max(0.0, 0.5 * (1.0 - (current_step / decay_steps)))
        self.action_offset[0] = new_bias


class Critic_Actor(nn.Module):
    """Red neuronal del Crític"""
    def __init__(self, global_state_dim):
        """Constructor de la red.
                Consiste en una cabeza para el estado de los vehiculos
                Otra para los vehiculos detectados
                Y una red fusión."""
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
        """Step por la red"""

        #aseguramos y evitamos que los tensores vengan mal dimensionados
        if global_state.dim() >= 3:
            global_state = global_state.squeeze(1)
        if cam_state.dim() == 3:
            cam_state = cam_state.squeeze(1)

        features = self.net(global_state)
        camera = self.camera_encoder(cam_state)
        fusion = torch.cat([features, camera], dim=-1)
        fusion = self.fusion(fusion)
        
        return fusion