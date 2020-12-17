import cv2 as cv
import os
import glob
import numpy as np
from sklearn.model_selection import train_test_split


def stack():
    lists = ['anger', 'disgust', 'fear', 'happy', 'sadness', 'surprise', 'contempt']
    actual_list = [os.path.join('CK+small', lists[i]) for i in range(len(lists))]
    image = []
    label = []
    for i in range(7):
        temp_img, temp_label = readImage(actual_list[i], i)
        image.append(np.array(temp_img))
        label.append(np.array(temp_label))
    images = image[0]
    for j in range(1, 7):
        images = np.concatenate((images, image[j]), axis=0)
    labels = np.hstack((label[0], label[1], label[2], label[3], label[4], label[5], label[6]))
    train_imgs_data, val_imgs_data,train_labels_data , val_labels_data = train_test_split(images, labels, test_size=60)
    return train_imgs_data, train_labels_data, val_imgs_data, val_labels_data


def readImage(path, count):
    images = []
    labels = []
    path = path + '/*.png'
    f = glob.glob(path)
    for file in f:
        image = cv.imread(file)
        image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        images.append(image)
        labels.append(count)
    return images, labels


if __name__ == '__main__':
    dir = "CK+small"
    train_imgs_data, train_labels_data, val_imgs_data, val_labels_data = stack()
    if not os.path.exists('./numpy'):
        os.makedirs('./numpy')
    np.save('./numpy/train_images.npy', train_imgs_data)
    np.save('./numpy/train_labels.npy', train_labels_data)
    np.save('./numpy/test_images.npy', val_imgs_data)
    np.save('./numpy/test_labels.npy', val_labels_data)




