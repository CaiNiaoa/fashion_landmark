import numpy as np
import cv2
import pandas as pd
import os

TRAIN_IMAGE_PATH = './dataset'

def load_csv(path):
    '''load .csv to ndarray'''
    return np.array(pd.read_csv(path))

def load_image(image_id, filename):
    '''brief:load image by image name'''
    #hedui
    image_path = os.path.join(TRAIN_IMAGE_PATH, filename, image_id)

    img =  cv2.imread(image_path)
    assert (img.all() != None)

    if filename == 'test':
        img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC)

    img  = img / 255.0
    assert (0<=img).all() and (img<=1.0).all()

    return img

