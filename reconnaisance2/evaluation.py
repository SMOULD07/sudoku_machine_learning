import torch 
import torch.nn as nn
import torch.nn.functional as F

def evaluate_model(model, test_loader, device):
    model.eval()  # Met le modèle en mode évaluation
    correct = 0
    total = 0

    with torch.no_grad():  # Désactive le calcul des gradients pour l'évaluation
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print("accuracy du modele : ", str(accuracy))