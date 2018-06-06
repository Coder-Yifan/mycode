from keras.models import load_model
from keras.preprocessing.image import img_to_array,load_img
import os
import cv2
import numpy as np

def predict_1(img_path,model_path,width,height):
    model = load_model(model_path)
    img = load_img(img_path, target_size=(width, height))
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
    result = model.predict(x)[0]
    y = np.max(result)
    label = str(np.where(result==y)[0])
    print('Predicted:{0}'.format(label) )
    print('real:{0}'.format(img_path))
def predict_more()
if __name__ == "__main__":
    img_path = 'E:/data/traffic/traffic-sign/test/00060/00552_00000.png'
    model_path = 'E:/data/traffic/mymodel.h5'
    predict_more(img_path,model_path,32,32)