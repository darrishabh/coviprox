from typing import Counter
import cv2
import numpy as np
import dlib
from PIL import Image
import math

def find_distances(face_list, coordinates_list,  AVG_WID = 18):
#     scale_percent = 300
#     width = int(img.shape[1] * scale_percent / 100)
#     height = int(img.shape[0] * scale_percent / 100)

# # dsize
#     dsize = (width, height)

# # resize image
#     img = cv2.resize(img, dsize)
#     width = img.shape[2]

#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     detector = dlib.get_frontal_face_detector()
#     predictor = dlib.shape_predictor("/home/angshuk/Desktop/models/shape_predictor_68_face_landmarks.dat")
#     faces = detector(gray)

    scales = []
    face_cent = []
    Counter = 0 
    
    for coord in coordinates_list:
        facewidth = coord[2]
        scales.append(AVG_WID/facewidth)
        
        x1=coord[0]
        y1=coord[1]
        x2= x1 + coord[2]
        y2= y1 + coord[3]
        
        centroid = (((x2-x1)/2)+x1, ((y2-y1)/2)+y1)
        faceimg = face_list[Counter]
        
        face_cent.append((faceimg, centroid))
        Counter +=1
    
    scale = sum(scales)/len(scales)
    
    distances = {}
    for i in range(len(face_cent)):
        for j in range(i+1, len(face_cent)):
            x1 = face_cent[i][1][0]
            x2 = face_cent[j][1][0]
            y1 = face_cent[i][1][1]
            y2 = face_cent[j][1][1]
            dist = math.sqrt(abs(math.pow((x2-x1),2)-math.pow((y2-y1),2)))*scale
            distances[str(i+1)+'-'+str(j+1)] = dist
    return distances