import cv2 as cv
import os
import glob
import numpy as np
from sklearn.model_selection import train_test_split

def makeDataSet(path, label):
    images = []
    labels = []
    path = path + '/*.png'
    for file in glob.glob(path):
        img = cv.imread(file, cv.IMREAD_GRAYSCALE)
        images.append(img)
        labels.append(label)
    return np.array(images, dtype = 'uint8'), np.array(labels, dtype = 'int64')

def createDataSet():
    lists = ['anger', 'disgust', 'fear', 'happy', 'sadness', 'surprise', 'contempt']
    actual_list = [os.path.join('CK+small', lists[i]) for i in range(len(lists))]
    anger_imgs, anger_labels = makeDataSet(actual_list[0], 0)
    comtempt_imgs, comtempt_labels = makeDataSet(actual_list[1], 1)
    disgust_imgs, disgust_labels = makeDataSet(actual_list[2], 2)
    fear_imgs, fear_labels = makeDataSet(actual_list[3], 3)
    happy_imgs, happy_labels = makeDataSet(actual_list[4], 4)
    sad_imgs, sad_labels = makeDataSet(actual_list[5], 5)
    surprise_imgs, surprise_labels = makeDataSet(actual_list[6], 6)

    imgs_data = np.vstack((anger_imgs, comtempt_imgs, disgust_imgs,
                           fear_imgs, happy_imgs, sad_imgs, surprise_imgs))
    labels_data = np.hstack((anger_labels, comtempt_labels, disgust_labels, fear_labels, happy_labels, sad_labels,
                             surprise_labels))


    train_imgs_data, val_imgs_data,train_labels_data , val_labels_data = train_test_split(imgs_data, labels_data, test_size=60)
    return train_imgs_data, train_labels_data, val_imgs_data, val_labels_data




if __name__ == '__main__':
    dir = "CK+small"
    train_imgs_data, train_labels_data, val_imgs_data, val_labels_data = createDataSet()
    if not os.path.exists('./numpy_data'):
        os.makedirs('./numpy_data')
    np.save('./numpy_data/train_images.npy', train_imgs_data)
    np.save('./numpy_data/train_labels.npy', train_labels_data)
    np.save('./numpy_data/val_images.npy', val_imgs_data)
    np.save('./numpy_data/val_labels.npy', val_labels_data)




