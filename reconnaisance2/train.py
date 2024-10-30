import torch 
import torch.nn as nn
import torch.nn.functional as F

def ftrain(model, device, train_loader, optimizer, epoch):
    model.train()  # Met le modèle en mode entraînement
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)  # Negative Log-Likelihood Loss
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:  # Afficher le log toutes les 10 itérations
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')
