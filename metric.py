import torch 
import numpy as np
from sklearn.metrics import f1_score

def accuracy(pred, label):
    count = 0
    for i in range(pred.shape[0]):
        if torch.argmax(torch.exp(pred[i])) == label[i]:
            count+=1
    acc = (count/pred.shape[0]) * 100 
    return acc

def cosine_distance(query, gallery):
    similarity_list = []
    cos = torch.nn.CosineSimilarity(dim = 0)
    for i in gallery:
        similarity = cos(query, i)
        similarity_list.append(similarity)
    similarity_array = torch.stack(similarity_list)
    return similarity_array

def f1_score(true_arr, pred_arr):
    y_true = true_arr
    pred_arr_new =  pred_arr
    f1 = f1_score(y_true, pred_arr_new)
    return f1