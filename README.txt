420_project - Facial Expression Recognition

Running process: make sure you have the CK+ file
- run face_detect
    It detects faces in images of CK+ dataset and generates a new directory named CK+small
- run preprocess
    It splits dataset into train, test and validation
    Resulting data will be stored in numpy file as numpy array
- run train_process
    The actual training process

Note:
    Face detection methods can be chosen from Haar or Hog
    Deep Learning model can be chosen from simple CNN or ResNet based CNN
