import torch
import pytest
import sys
import os

# Ajout du chemin src pour l'import
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from model import get_model

def test_model_output_shape():
    """
    Vérifie que la dimension de sortie du modèle correspond au nombre de classes.
    """
    num_classes = 38
    model = get_model(num_classes=num_classes)
    dummy_input = torch.randn(1, 3, 224, 224)
    output = model(dummy_input)
    
    assert output.shape == (1, num_classes), f"Erreur: Forme de sortie attendue (1, {num_classes}), reçue {output.shape}"

def test_model_forward_pass():
    """
    Vérifie qu'un forward pass ne produit pas de NaN.
    """
    model = get_model(num_classes=10)
    dummy_input = torch.randn(2, 3, 224, 224)
    output = model(dummy_input)
    
    assert not torch.isnan(output).any(), "Erreur: Le modèle produit des NaN"
