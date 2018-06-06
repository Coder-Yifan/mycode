import sys
import random
import cv2
from keras.preprocessing.image import ImageDataGenerator,img_to_array
from keras.utils import to_categorical
import glob
import os
import glob
import numpy as np

def load_data(path,width,height,class_num):
    data = []
    labels = []
    FileList = glob.glob(path+'/*')
    imagepaths = []
    for file in FileList:
        image = glob.glob(file + '/*.png')
        imagepaths+=image
    random.seed(42)
    random.shuffle(imagepaths)
    for imagepath in imagepaths:
        image = cv2.imread(imagepath)
        image = cv2.resize(image, (width, height))
        image = img_to_array(image)
        data.append(image)
        label = int(imagepath.split(os.path.sep)[-2])
        labels.append(label)
    data = np.array(data, dtype="float") / 255.0
    labels = np.array(labels)
    labels = to_categorical(labels, num_classes=class_num)
    return data, labels
