# -*- coding: utf-8 -*-

import dlib
import numpy
import glob
from skimage import io
import numpy  as np
predictor_path = "./shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

lst = glob.glob('./datasets0/*.jp*')



def get_landmarks_m(path):
    im = io.imread(path)
    dets = detector(im, 1)

    print("Number of faces detected: {}".format(len(dets)))

    for i in range(len(dets)):

        facepoint = np.array([[p.x, p.y] for p in predictor(im, dets[i]).parts()])

        for i in range(68):
            x=facepoint[i][1]
            y=facepoint[i][0]
            im[x-3:x+3,y-3:y+3,1] = 255
    return im

for i in lst:
    name = i.split('/')[-1]
    print("face_landmark:"+name)
    io.imsave('result/'+name,get_landmarks_m(i))
