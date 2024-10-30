import torch 
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5)  
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5)
        self.dropout = nn.Dropout2d()
        self.fc1 = nn.Linear(1024, 128)  
        self.fc2 = nn.Linear(128, 10)  # 10 sorties pour 10 classes (chiffres de 0 à 9)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))  
        x = F.relu(F.max_pool2d(self.dropout(self.conv2(x)), 2))
        x = x.view(-1, 1024)  #aplatir les données 
        x = F.relu(self.fc1(x))
        x = self.dropout(x) #desactivation de certain neurones au hasard 
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

