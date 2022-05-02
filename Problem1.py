import numpy as np
import os
from sklearn.model_selection import train_test_split
from keras.preprocessing import image
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder

def load_data_grayscale():
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
def load_data_rgb():
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
            img = image.load_img(img_path+'/'+str(f), grayscale=False,color_mode='rgb', target_size=(50, 50))

            img_array = image.img_to_array(img)
            img_array[0][0][0] = img_array[0][0][0]* img_array[0][0][1]*img_array[0][0][2]
            
            for j in range(0,50):
                for k in range(0,50):
                    img_array[j][k][0] = img_array[j][k][0] * img_array[j][k][1] * img_array[j][k][2]
            img_array = img_array[:,:,0]
            images.append(img_array)
            lb = x
            labels.append(lb);
    data = np.array(images)
    labels = np.array(labels)

    return data, labels
def predict_grayscale(n):
    (x_train, x_test, y_train, y_test) = train_test_split(images_grayscale_new, encoder_labels, test_size=0.2)
    accuracy = []
    for i in range(1, n):
        knn = KNeighborsClassifier(n_neighbors=i)
        knn.fit(x_train,y_train)
        y_pred = knn.predict(x_test)
        accuracy_value = metrics.accuracy_score(y_test,y_pred)
        accuracy.append(accuracy_value)
    accuracy = np.array(accuracy)
    max_n_neighbors = np.array(np.where(accuracy==max(accuracy)))
    min_n_neighbors = np.array(np.where(accuracy==min(accuracy)))
    print("(a) 像素級灰度表示---  最大索引值 : {} ，當前最大值 : {}".format(max_n_neighbors,max(accuracy)))
    print("(a) 像素級灰度表示---  最小索引值 : {} ，當前最小值 : {}".format(min_n_neighbors,min(accuracy)))
def predict_rgb(n):
    (x_train, x_test, y_train, y_test) = train_test_split(images_rgb_new, encoder_labels, test_size=0.2)
    accuracy = []
    for i in range(1, n):
        knn = KNeighborsClassifier(n_neighbors=i)
        knn.fit(x_train,y_train)
        y_pred = knn.predict(x_test)
        accuracy_value = metrics.accuracy_score(y_test,y_pred)
        accuracy.append(accuracy_value)
    accuracy = np.array(accuracy)
    max_n_neighbors = np.array(np.where(accuracy==max(accuracy)))
    min_n_neighbors = np.array(np.where(accuracy==min(accuracy)))
    print("(a) 像素級顏色表示---  最大索引值 : {} ，當前最大值 : {}".format(max_n_neighbors,max(accuracy)))
    print("(a) 像素級顏色表示---  最小索引值 : {} ，當前最小值 : {}".format(min_n_neighbors,min(accuracy)))


print("Loading data...")
images_grayscale, labels = load_data_grayscale()
images_rgb, labels = load_data_rgb()
images_grayscale_new = np.array(images_grayscale).reshape(1000,-1)
images_rgb_new = np.array(images_rgb).reshape(1000,-1)
labelencoder = LabelEncoder()
encoder_labels = labelencoder.fit_transform(labels) #進行Labelencoding編碼
#print(encoder_labels)

predict_grayscale(50)
predict_rgb(50)
