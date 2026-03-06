import torch
import torch.nn as nn
import torchvision.models as models

class PlantClassifier(nn.Module):
    def __init__(self, num_classes=10):
        super(PlantClassifier, self).__init__()
        # Utilisation de ResNet50 pré-entraîné
        self.backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        
        # Remplacement de la dernière couche pour notre nombre de classes
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)

def get_model(num_classes, device='cpu'):
    model = PlantClassifier(num_classes=num_classes)
    return model.to(device)
