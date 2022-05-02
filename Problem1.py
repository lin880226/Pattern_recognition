import numpy as np
from keras.utils import np_utils  # 用來後續將 label 標籤轉為 one-hot-encoding
from matplotlib import pyplot as plt
from keras.models import Sequential, model_from_yaml, load_model
from keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from keras.optimizers import Adam, SGD
import os
from sklearn.model_selection import train_test_split
from keras.preprocessing import image
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import cv2

def load_data():
    path = './TenCategories/'
    files = os.listdir(path)
    images = []
    labels = []
    for x in files:
        path = './TenCategories/'
        path  = path + x
        files = os.listdir(path)
        for f in files:
            img_path = path 
            # print(img_path)
            # print(img_path+'/'+str(f))
            img = image.load_img(img_path+'/'+str(f), grayscale=True, target_size=(50, 50))
            img_array = image.img_to_array(img)
            images.append(img_array)
    
            lb = x
            labels.append(lb);
    data = np.array(images)
    labels = np.array(labels)

    return data, labels

print("Loading data...")
images, lables = load_data()


(x_train, x_test, y_train, y_test) = train_test_split(images, lables, test_size=0.2)

