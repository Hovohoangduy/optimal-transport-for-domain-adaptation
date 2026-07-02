import torch
import torch.nn as nn

class FeatureExtractor(nn.Module):
    def __init__(self, latent_dim=256, input_size=(28, 28)):
        super(FeatureExtractor, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=5, stride=1, padding=2),
        )