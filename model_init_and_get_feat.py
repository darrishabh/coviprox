from REID.model import make_model
import torch

def get_model(cfg):
    model = make_model(cfg, 10, 1, 1)
    model.state_dict(torch.load('/home/angshuk/Desktop/angshukdutta10-projects-reid-20-output/transformer_40.pth'))
    model.eval()
    return model

def get_data_output(model, query, gallery):
    combined_data = torch.cat([query, gallery])
    feat = model(combined_data)
    return feat
