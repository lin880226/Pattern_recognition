import numpy as np
import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
import cv2
from keras.preprocessing import image
from sklearn.model_selection import cross_val_score
def load_data_global_color_histogram(a,z):
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
            #print(img_path)
            #print(img_path+'/'+str(f))
            img = image.load_img(img_path+'/'+str(f), grayscale=False,color_mode='rgb',target_size=(50, 50))
            i=0
            w = a
            hight, width = img.size
            while (i + w <= hight):
                j = 0
                while (j + w <= width):
                    new_img = img.crop((i, j, i + w, j + w))
                    j += 1  #滑动步长
                    new_img = np.array(new_img)
                    (B, G, R) = cv2.split(new_img)
                    hb = cv2.calcHist([B], [0], None, [z], [0,256])
                    hg = cv2.calcHist([G], [0], None, [z], [0,256])
                    hr = cv2.calcHist([R], [0], None, [z], [0,256])
                    img_array = np.c_[hb,hg,hr]
                    images.append(img_array)
                    
                i = i + 1
            lb = x
            labels.append(lb);
    data = np.array(images)
    labels = np.array(labels)
    return data, labels

def predict_global_color_histogram():

    k_range = range(1,31)
    bin_range = range(10,40,10)
    w_range  = range(1,5,1)
    k_scores = []
    for k_number in k_range:
        for bin_number in bin_range:
            for w in w_range:
                print("Loading data...")
                images_global_color_histogram, labels = load_data_global_color_histogram(w,bin_number)
                images_global_color_histogram_new = np.array(images_global_color_histogram).reshape(1000,-1)
                labelencoder = LabelEncoder()
                encoder_labels = labelencoder.fit_transform(labels) #進行Labelencoding編碼
                knn = KNeighborsClassifier(n_neighbors=k_number)
                knn.fit(images_global_color_histogram_new,encoder_labels)
                scores = cross_val_score(knn,images_global_color_histogram_new,encoder_labels,cv=10,scoring='accuracy')
                k_scores.append(scores.mean())
            

predict_global_color_histogram()