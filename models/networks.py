import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np


class Actor_network(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()

        self.register_buffer("img_mean", torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1))
        self.register_buffer("img_std", torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1))
        
        #carrgamos ResNet18 preentrenado
        resnet = models.resnet18(pretrained=True)
        #quitamos la última cap para quedarnos con el vector de 512
        self.encoder = nn.Sequential(*list(resnet.children())[:-1])
        
        self.encoder.eval()
        for p in self.encoder.parameters():
            p.requires_grad = False
        
        #reducimos la imagen de 512 a 128 dimensiones. 
        self.visual_proj = nn.Sequential(
            nn.Linear(512, 128),
            nn.LayerNorm(128),
            nn.ReLU()
        )

        self.state_net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.LayerNorm(64),
            nn.ReLU()
        )

        self.fusion_net = nn.Sequential(
            nn.Linear(192, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )

        self.mean_layer = nn.Linear(128, action_dim)
        
        with torch.no_grad():
            self.mean_layer.bias[0] = 1.0   
            self.mean_layer.bias[1] = 0.0  

        self.std_layer = nn.Linear(128, action_dim)

    def forward(self, image, state):
        image = image / 255.0 
        image = (image - self.img_mean) / self.img_std
        
        
        with torch.no_grad():
            visual_features = self.encoder(image)
            visual_features = torch.flatten(visual_features, 1) 
        
        visual_emb = self.visual_proj(visual_features) 

        state_emb = self.state_net(state) 

        fused = torch.cat([visual_emb, state_emb], dim=1) 
        features = self.fusion_net(fused)

        mean = self.mean_layer(features)
        log_std = self.std_layer(features)
        std = torch.exp(torch.clamp(log_std, -2, 1))

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
    
