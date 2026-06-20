import torch
import torch.nn as nn
import numpy as np


class SegEncoder(nn.Module):
    """CNN ligera que toma class IDs (B,H,W) de CARLA semantic_segmentation
    y devuelve features (B, out_dim). One-hot interno sobre las clases relevantes."""
    # CARLA tags: 4=Pedestrian, 6=RoadLine, 7=Road, 8=SideWalk, 10=Vehicles, 18=TrafficLight
    SELECTED_CLASSES = [1, 2, 7, 12, 14, 24]

    def __init__(self, out_dim=128, img_size=128):
        super().__init__()
        K = len(self.SELECTED_CLASSES)
        self.register_buffer(
            'classes',
            torch.tensor(self.SELECTED_CLASSES, dtype=torch.long).view(1, -1, 1, 1)
        )
        self.conv = nn.Sequential(
            nn.Conv2d(K, 32, kernel_size=8, stride=4),   # 128 -> 31
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),  # 31  -> 14
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),  # 14  -> 12
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 12 * 12, out_dim),
            nn.ReLU(),
        )

    def forward(self, image_ids):
        # image_ids: (B, H, W) long con class IDs 0-22
        oh = (image_ids.unsqueeze(1) == self.classes).float()  # (B, K, H, W)
        return self.conv(oh)


class Actor_network(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.encoder = SegEncoder(out_dim=128)

        self.state_net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.LayerNorm(64),
            nn.ReLU()
        )

        self.fusion_net = nn.Sequential(
            nn.Linear(128 + 64, 128),
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
        # image: (B, H, W) long
        visual_emb = self.encoder(image)
        state_emb = self.state_net(state)
        fused = torch.cat([visual_emb, state_emb], dim=1)
        features = self.fusion_net(fused)

        mean = self.mean_layer(features)
        log_std = self.std_layer(features)
        std = torch.exp(torch.clamp(log_std, -2, 1))
        return mean, std


class Critic_Actor(nn.Module):
    def __init__(self, state_dim, num_agents):
        super().__init__()
        self.num_agents = num_agents
        self.encoder = SegEncoder(out_dim=128)

        self.state_net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU()
        )

        self.fusion = nn.Sequential(
            nn.Linear(128 + num_agents * 128, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, state, images):
        # images: list de tensores (B, H, W) long (uno por agente) o tensor (B, A, H, W) long
        if isinstance(images, (list, tuple)):
            images = torch.stack(images, dim=1)
        B, A, H, W = images.shape
        imgs_flat = images.view(B * A, H, W)
        vis = self.encoder(imgs_flat)            # (B*A, 128)
        vis = vis.view(B, A * 128)
        
        if st.dim() >= 3:
            st = st.squeeze(1)
        st = self.state_net(state)
        fused = torch.cat([st, vis], dim=1)
        return self.fusion(fused)

    
