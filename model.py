import torch 
from torch import nn 
import torch.nn.functional as F 
import torchvision

class finetune_model(nn.Module):
    def __init__(self):
        super(finetune_model, self).__init__()
        self.lin1 = nn.Linear(25088, 4096)
        self.drop1 = nn.Dropout(p = 0.5)
        self.lin2 = nn.Linear(4096, 128)
        self.lin3 = nn.Linear(128, 2)
        self.logsoftmax1 = nn.LogSoftmax(dim=1)

        
    def forward(self, x):
        x = F.relu(self.drop1(self.lin1(x)))
        x = F.relu(self.lin2(x))
        x = self.lin3(x)
        x = self.logsoftmax1(x)
        return x

#25088,  4096