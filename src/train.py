import torch
import torch.nn as nn
import torch.optim as optim
from model import get_model
from data_utils import set_seed, get_transforms, PlantDataset
from torch.utils.data import DataLoader

def train():
    # 1. Setup
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Entraînement sur : {device}")

    # 2. Data (Simulée pour l'exemple)
    train_transform, val_transform = get_transforms()
    dummy_paths = ["path/to/img.jpg"] * 100
    dummy_labels = [i % 10 for i in range(100)]
    
    train_ds = PlantDataset(dummy_paths, dummy_labels, transform=train_transform)
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)

    # 3. Model, Loss, Optimizer
    model = get_model(num_classes=10, device=device)
    criterion = nn.CrossEntropyLoss() # Fonction de perte minimisée
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4) # Regularization: L2 (weight_decay)

    # 4. Training Loop (abrégée pour la démo)
    model.train()
    for epoch in range(1):
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            if i % 5 == 0:
                print(f"Epoch 1, Batch {i}, Loss: {loss.item():.4f}")

    print("Entraînement terminé.")
    torch.save(model.state_dict(), "plant_model_v1.pth")

if __name__ == "__main__":
    train()
