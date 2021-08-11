import torch
import torchvision
import os
import cv2
from model_init_and_get_feat import get_data_output
from metric import cosine_distance

def do_inference(num_query, feat, Gal_ID_dict):
    id_dict = {}
    for i in range(num_query):
        similarity = cosine_distance(feat[i], feat[num_query:])
        id = Gal_ID_dict[torch.argmax(similarity).item() + 1]
        id_dict[i+1] = id 
    return id_dict   

def get_ID(gal_path, model, query_list):
    
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToPILImage(), 
        torchvision.transforms.Resize(256),
        torchvision.transforms.ToTensor()
    ])
    gallery_dict = {}; gal_counter = 1; gallery_feature_list = []
    
    for i in range(len(query_list)):
        query_list[i] = transform(query_list[i])
    
    query = torch.stack(query_list)

    for file in os.listdir(gal_path):
        gallery_img = cv2.imread(os.path.join(gal_path, file))
        gallery_img = transform(gallery_img)
        gallery_feature_list.append(gallery_img)
        gallery_dict[gal_counter] = file
        gal_counter+=1
    
    gallery_feature_tens = torch.stack(gallery_feature_list)
    feat = get_data_output(model, query, gallery_feature_tens)
    id_dict = do_inference(query.shape[0], feat, gallery_dict)

    return id_dict
