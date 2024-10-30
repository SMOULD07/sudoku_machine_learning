import torch 
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import model as mod 
import train 
import evaluation 
from PIL import Image




# Définissez les transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)) # normalisation de la base de données 
])

# Téléchargement du dataset MNIST
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# creation des data loader 
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)






                    ###############################################################
                                            #entrainement


model = mod.Net().to("cpu")
optimizer = torch.optim.Adam(model.parameters())

for epoch in range(5):  # Nombre d'époques
    train.ftrain(model,"cpu", train_loader, optimizer, epoch)

# sauvgarder le modele
torch.save(model.state_dict(), 'mnist_model.pth')
print("Modèle sauvegardé sous 'mnist_model.pth'")

                    ###############################################################
                                            # evaluation


device = "cpu"
model.to(device)
evaluation.evaluate_model(model, test_loader, device)




                    ################################################################
                                        # test 

def test(image,model):
    model.eval()
    with torch.no_grad():
        output=model(image)
        _,predicted=torch.max(output,1)
        print("prediction ============> ",predicted[0].item())








transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # Convertit en niveaux de gris
    transforms.Resize((28, 28)),  # Redimensionne à 28x28
    transforms.ToTensor(),  # Convertit en tenseur PyTorch
    transforms.Normalize((0.5,), (0.5,))  # Normalise les valeurs des pixels
])

def load_image(image_path):
    image = Image.open(image_path)
    image = transform(image)
    return image




# image = load_image("C:/Users/utilisateur/Desktop/projet/images/im.PNG")
# test(image, model)