import torchvision
import torchvision.models as models
import torch
import torchreid
from torchreid.data.datasets import image

def get_pretrained_model(ans = True):
    if ans==True:
        resnet_mod = models.resnet50(pretrained=True)
        torch.save(resnet_mod, '/home/angshuk/Desktop/TransReID/model/backbones/resnet50.pt')
    else:
        vit_transreid = torch.load('/home/angshuk/Downloads/vit_transreid_duke.pth')
        print(vit_transreid)



get_pretrained_model(False)