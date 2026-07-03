import torch
import torch.nn as nn
import torch.optim as optim
import os, sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))
from utils.mnist_usps_dataloader import get_data_loaders
from feature_extractor import FeatureExtractor

def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epochs = 10
    lr = 0.001

    print("Source Training!!!")

    source_loader, _ = get_data_loaders(batch_size=32)

    # model
    backbone = FeatureExtractor(
        latent_dim=256,
        input_size=(28, 28)
    ).to(device)

    classifier_head = nn.Linear(256, 10).to(device)

    optimizer = optim.Adam(list(backbone.parameters()) + list(classifier_head.parameters()), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # training loop
    for epoch in range(epochs):
        backbone.train()
        total_loss = 0
        correct = 0
        total = 0

        for data, target in source_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = classifier_head(backbone(data))
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
        
        print(f"Epoch {epoch+1}/{epochs} | Accuracy: {100 * correct / total:.2f}%")
    
    # save model
    save_path = "saved_models/mnist_extractor.pth"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(backbone.state_dict(), save_path)
    print(f"Model saved: {save_path}")

if __name__=="__main__":
    train_model()