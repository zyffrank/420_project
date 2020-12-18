import cv2
import numpy as np
import glob
from sklearn.model_selection import train_test_split
import os


# distribute the images set to train and validation set
def stack():
    # use the default type to read images and label images
    lists = ['anger', 'disgust', 'fear', 'happy', 'sadness', 'surprise', 'contempt']
    actual_list = [os.path.join('CK+small', lists[i]) for i in range(len(lists))]
    image = []
    label = []
    for i in range(7):
        temp_img, temp_label = readAndLabel(actual_list[i], i)
        image.append(np.array(temp_img))
        label.append(np.array(temp_label))
    images = image[0]
    for j in range(1, 7):
        images = np.concatenate((images, image[j]), axis=0)
    labels = np.hstack((label[0], label[1], label[2], label[3], label[4], label[5], label[6]))
    training, test, training_labels, test_labels = train_test_split(images, labels, test_size=20)
    return training, training_labels, test, test_labels


# label all images from 0 to 6
# study from https://stackoverflow.com/questions/49537604/how-to-read-multiple-images-from-multiple-folders-in-python
def readAndLabel(file_name, type):
    folders = glob.glob(file_name)
    emotion_image = []
    emotion_label = []
    for folder in folders:
        for file in glob.glob(folder + '/*.png'):
            image = cv2.imread(file)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            emotion_image.append(image)
            emotion_label.append(type)
    return emotion_image, emotion_label


if __name__ == '__main__':
    if not os.path.exists('./numpy'):
        os.makedirs('./numpy')
    training, training_labels, test, test_labels = stack()
    np.save('./numpy/train_images.npy', training)
    np.save('./numpy/train_labels.npy', training_labels)
    np.save('./numpy/test_images.npy', test)
    np.save('./numpy/test_labels.npy', test_labels)




