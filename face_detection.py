import glob
import cv2
from PIL import Image
import os
import dlib

# Face detection method, use Haar or HoG
def detect_face(path, cas_model):
    path = path + '/*.png'
    hogFaceDetector = dlib.get_frontal_face_detector()
    for file in glob.glob(path):
        img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
        # faces = cas_model.detectMultiScale(img)
        # scaleFactor: Parameter specifying how much the image size is reduced at each image scale.
        # Parameter specifying how many neighbors each candidate rectangle should have to retain it
        faces = hogFaceDetector(img, 1)
        for (i, rect) in enumerate(faces):
            x = rect.left()
            y = rect.top()
            w = rect.right() - x
            h = rect.bottom() - y
            img = img[y:y + h, x:x + w]
            break
        # for (x, y, w, h) in faces:
        #     img = img[y:y + h, x:x + w]
        #     break

        img = cv2.GaussianBlur(img, (9, 9), 0)
        img = cv2.resize(img, (50, 50))
        save_path = file.replace('CK+', 'CK+small')
        Image.fromarray(img, 'L').save(save_path)


if __name__ == '__main__':
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    # Create directory for our detected small image
    if not os.path.exists('./CK+small'):
        os.makedirs('./CK+small')
        os.makedirs('./CK+small/anger')
        os.makedirs('./CK+small/contempt')
        os.makedirs('./CK+small/disgust')
        os.makedirs('./CK+small/fear')
        os.makedirs('./CK+small/happy')
        os.makedirs('./CK+small/sadness')
        os.makedirs('./CK+small/surprise')
    dir = 'CK+'
    anger_path = os.path.join(dir, 'anger')
    disgust_path = os.path.join(dir, 'disgust')
    fear_path = os.path.join(dir, 'fear')
    happy_path = os.path.join(dir, 'happy')
    sadness_path = os.path.join(dir, 'sadness')
    surprise_path = os.path.join(dir, 'surprise')
    contempt_path = os.path.join(dir, 'contempt')

    print("processing anger")
    detect_face(anger_path, face_cascade)
    print("processing contempt")
    detect_face(contempt_path, face_cascade)
    print("processing disgust")
    detect_face(disgust_path, face_cascade)
    print("processing fear")
    detect_face(fear_path, face_cascade)
    print("processing happy")
    detect_face(happy_path, face_cascade)
    print("processing sadness")
    detect_face(sadness_path, face_cascade)
    print("processing surprise")
    detect_face(surprise_path, face_cascade)

