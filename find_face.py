import cv2
import torch
import numpy as np
import torchvision

def get_faces(path):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    face_dict = {}
    coordinate_dict = {}
    face_img = cv2.imread(path)
    gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    face_loc = face_cascade.detectMultiScale(gray, 1.3, 5)
    count = 1
    
    for f_loc in face_loc:
        x, y, w, h = [ v for v in f_loc ]
        faces = face_img[y:y+h, x:x+w]
        faces = cv2.cvtColor(faces, cv2.COLOR_BGR2RGB)
        face_dict[count] = faces; coordinate_dict[count] = [x, y, w, h]
        count+=1
    return face_dict, coordinate_dict 

def get_mask_prediction(face_list, model):
    face_tens_pred_dict = {}
    face_tens = torch.stack(face_list)
    face_tens_logits = model(face_tens)
    face_tens_pred = torch.argmax(torch.exp(face_tens_logits), dim=1)
    
    for i in range(face_tens_pred.shape[0]):
        face_tens_pred_dict[i+1] = face_tens_pred[i]

    return face_tens_pred_dict
