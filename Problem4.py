import numpy as np
import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
import cv2
from keras.preprocessing import image
from sklearn.model_selection import cross_val_score
from skimage.feature import hog

def load_data(size):
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
            img = image.load_img(img_path+'/'+str(f), grayscale=False,color_mode='rgb', target_size=(size, size))
            img_array = image.img_to_array(img)
            images.append(img_array)
            lb = x
            labels.append(lb);
    data = np.array(images)
    labels = np.array(labels)
    return data, labels


def predict():
    i=0
    k_scores = []
    max_value = np.array([[0,0,0,0,0],[0,0,0,0,0]])
    temp = 0
    for k_number in range(2,10,1):
        for img_size in range(50,250,100):
            for cell_size in range(10,40,20):
                for orientations in range(4,14,2):
                    i+=1
                    images, labels = load_data(img_size)
                    images_new = np.array(images).reshape(1000,-1)
                    labelencoder = LabelEncoder()
                    encoder_labels = labelencoder.fit_transform(labels) #進行Labelencoding編碼
                    hog_image = hog(images_new, orientations=orientations, pixels_per_cell=(cell_size, cell_size),cells_per_block=(2, 2), visualize=True)
                    hog_image_array = np.array(hog_image)
                    hog_image_array = hog_image_array[1]
                    
                    knn = KNeighborsClassifier(n_neighbors=k_number)
                    knn.fit(hog_image_array,encoder_labels)
                    scores = cross_val_score(knn,hog_image_array,encoder_labels,cv=10,scoring='accuracy')
                    k_scores.append(scores.mean())
                    print(i)
                    print('Fold: {},img_size: {},cell_size: {} , orientations:{},Training/Test Split Distribution: {}, Accuracy: {}' .format(k_number+1,img_size, cell_size,orientations,np.bincount(encoder_labels), k_scores))
                    max_ = max(k_scores)
                    max_list = np.array([k_number,img_size,cell_size,orientations,max_])
                    if temp  == max_:  
                        max_value = np.vstack([max_value,max_list])
                    elif temp < max_:
                        temp = max_
                        max_value = np.delete(max_value,-1,axis=0)
                        max_value = np.vstack([max_value,max_list])

    return max_value

predict()
