from torch.nn.modules import distance
from data_processing import get_transform_object
from os import name
import torch, torchvision
import cv2
import numpy as np
from probability import single_adjusted_probability,final_probability
from finddistance import *
from find_face import get_faces, get_mask_prediction
import argparse
from model import finetune_model
import cv2
from REID.config import  cfg
from model_init_and_get_feat import get_model
from inference import get_ID
                    
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='parser to get path for config file for model')
    parser.add_argument('--config_path', type = str, help = 'the path to the config file')
    parser.add_argument('--img_path', type = str, help = 'the path to the test image file')
    parser.add_argument('--mask_path', type = str, help = 'The Mask Detector model path')
    parser.add_argument('--gal_path', type = str, help = 'path to gallery images')
    args = parser.parse_args()
    
    if args.config_path != "":
        cfg.merge_from_file(args.config_path)
        cfg.freeze()
    
    
    face_list = []
    finetune_mod = finetune_model()
    transform = get_transform_object(False)
    
    mob_mod = torchvision.models.vgg19_bn(pretrained= False)
    mob_mod.classifier = finetune_mod
    mob_mod.load_state_dict(torch.load(args.mask_path))
    mob_mod.eval()
    reid_model = get_model(cfg)
    
    face_dict, coord = get_faces(args.img_path)

    for idx in range(len(face_dict.keys())):
        face_list.append(transform(face_dict[idx+1]))
    
    pred = get_mask_prediction(face_list, mob_mod)
    distance_dict = find_distances(list(face_dict.values()), list(coord.values()))   
    
    gal_path = args.gal_path
    id_of_faces_dict = get_ID(gal_path, reid_model, list(face_dict.values()))

    counter = 0; single_prob_dict = {} 

    for i in range(len(face_dict.keys())):
        for j in range(i+1, len(face_dict.keys())):
            idx = list(distance_dict.keys())[counter]
            single_prob = single_adjusted_probability(pred[i+1], pred[i+2], distance_dict[idx])
            single_prob_dict[idx] = single_prob
            counter+=1

    final_counter = 0; final_dict = {}
    for i in single_prob_dict.keys():
        idx = i.split(sep = '-')
        person_1_id = id_of_faces_dict[int(idx[0])]
        person_2_id = id_of_faces_dict[int(idx[1])]
        final_dict[person_1_id + '-' + person_2_id] = single_prob_dict[i]

    print('***************Printing Results******************')
    for i in final_dict.keys():
        prob = final_dict[i]
        idx1 = i.split(sep='-')[0]
        idx2 = i.split(sep='-')[1]
        print('| Person {} can contract COVID from Person {} with a likelihood of {:.2f} |'.format(idx1, idx2, prob))
    


    







    
    

   
    


