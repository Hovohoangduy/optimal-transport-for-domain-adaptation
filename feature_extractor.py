import torch
import torch.nn as nn

class FeatureExtractor(nn.Module):
    def __init__(self, latent_dim=256, input_size=(28, 28)):
        super(FeatureExtractor, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2) # output: N x 64 x W/2 x H/2
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=2), # output N x 64 X W X H
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2) # output: N x 128 x W/4 x H/4
        )
        dummy_tensor = torch.randn(1, 1, input_size[0], input_size[1])
        x = self.conv1(dummy_tensor)
        x = self.conv2(x)

        self.flattened_size = x.numel()
        self.fc = nn.Sequential(
            nn.Linear(self.flattened_size, latent_dim),
            nn.BatchNorm1d(latent_dim),
            nn.ReLU()
        )
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)

        # flatten
        x = x.view(x.size(0), -1)
        features = self.fc(x)
        return features
    
if __name__=="__main__":
    dummy_input = torch.randn(64, 1, 28, 28)
    model = FeatureExtractor()
    output = model(dummy_input)
    print(f"Input model: {dummy_input.shape}")
    print(f"Ouput model (Features): {output.shape}")