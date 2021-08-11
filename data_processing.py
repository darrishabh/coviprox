from torch.utils.data import Dataset
import torch
from torchvision import transforms
import numpy as np
import os
import cv2
from torchvision.transforms.transforms import CenterCrop, Resize


def create_data(path):
    for i in os.listdir(path):
        if i == 'train':
            train_imgs, train_labels = read_data(path, i)
        elif i == 'val':
            val_imgs, val_labels = read_data(path, i)
        else:
            test_imgs, test_labels = read_data(path, i)

    return train_imgs, train_labels, val_imgs, val_labels, test_imgs, test_labels 

        # for j in os.listdir(path+'/'+i):
        #     print(j)

def read_data(path, name):
    image_list = []; label_list = []
    new_path = os.path.join(path,name)
    for i in os.listdir(new_path):
        new_image_path = os.path.join(new_path, i)
        print(new_image_path)
        for j in os.listdir(new_image_path):
            image = cv2.imread(os.path.join(new_image_path, j))
            image = cv2.resize(image, (224,224))
            image_list.append(image)
            label_list.append(int(i))
        
 
    label_tens  = torch.LongTensor(label_list)
    image_tens = torch.transpose(torch.FloatTensor(image_list), 1, 3)
    return image_tens, label_tens


class create_dataset(Dataset):
    def __init__(self, train_datafile, label_file):
        super(create_dataset, self).__init__()
        self.data = train_datafile
        self.label = label_file
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return (self.data[idx], self.label[idx])


def get_transform_object(compose = True):
    if compose == True:
        transform = transforms.Compose([
                transforms.ToPILImage(), 
                transforms.Resize(256),
                transforms.RandomAffine(degrees = 0, shear = 0.15),
                transforms.RandomHorizontalFlip(p=0.5), 
                transforms.RandomRotation(20),
                transforms.RandomResizedCrop(224),
                transforms.ToTensor(), 
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
    else:
        transform = transforms.Compose([ 
            transforms.ToPILImage(), 
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(), 
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    
    return transform 
    