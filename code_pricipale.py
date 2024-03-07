

import cv2
import numpy
import math
import os
from keras.preprocessing.image import img_to_array, load_img
from keras.optimizers import SGD
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from PIL import Image
import keras

# path to the dataset
paths = ['/home/deo/Téléchargements/jeu de donnees']
TOTAL_DATASET = 2515
x_train = []  
y_train = []
nb_classes = 36  
img_rows, img_cols = 400, 400 
img_channels = 3  
batch_size = 32
nb_epoch = 100  
data_augmentation = True


classes = {
    '0': 0,
    '1': 1,
    '2': 2,
    '3': 3,
    '4': 4,
    '5': 5,
    '6': 6,
    '7': 7,
    '8': 8,
    '9': 9,
    'a': 10,
    'b': 11,
    'c': 12,
    'd': 13,
    'e': 14,
    'f': 15,
    'g': 16,
    'h': 17,
    'i': 18,
    'j': 19,
    'k': 20,
    'l': 21,
    'm': 22,
    'n': 23,
    'o': 24,
    'p': 25,
    'q': 26,
    'r': 27,
    's': 28,
    't': 29,
    'u': 30,
    'v': 31,
    'w': 32,
    'x': 33,
    'y': 34,
    'z': 35,
}


def load_data_set():
    for path in paths:
        for root, directories, filenames in os.walk(path):
            for filename in filenames:
                if filename.endswith(".jpeg"):
                    fullpath = os.path.join(root, filename)
                    img = load_img(fullpath)
                    img = img_to_array(img)
                    x_train.append(img)
                    t = fullpath.rindex('/')
                    fullpath = fullpath[0:t]
                    n = fullpath.rindex('/')
                    y_train.append(classes[fullpath[n + 1:t]])


def make_network(x_train):
    model = Sequential()

    model.add(Convolution2D(32, 3, 3, border_mode='same',
                            input_shape=x_train.shape[1:]))
    model.add(Activation('relu'))
    model.add(Convolution2D(32, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Convolution2D(64, 3, 3, border_mode='same'))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))

    return model


def train_model(model, X_train, Y_train):
   
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy',
                  optimizer=sgd,
                  metrics=['accuracy'])

    model.fit(X_train, Y_train,
              batch_size=batch_size,
              nb_epoch=nb_epoch)


def trainData():
    load_data_set()
    a = numpy.asarray(y_train)
    y_train_new = a.reshape(a.shape[0], 1)

    X_train = numpy.asarray(x_train).astype('float32')
    X_train = X_train / 255.0
    Y_train = np_utils.to_categorical(y_train_new, nb_classes)

    model = make_network(numpy.asarray(x_train))
    train_model(model,X_train,Y_train)
    model.save('/home/deo/Téléchargements/resulata')

  

    return model


model = trainData()



def identifyGesture(handTrainImage):
   
    handTrainImage = cv2.cvtColor(handTrainImage, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(handTrainImage)
    img_w, img_h = img.size
    M = max(img_w, img_h)
    background = Image.new('RGB', (M, M), (0, 0, 0))
    bg_w, bg_h = background.size
    offset = ((bg_w - img_w) / 2, (bg_h - img_h) / 2)
    background.paste(img, offset)
    size = 400,400
    background = background.resize(size, Image.ANTIALIAS)

    
    open_cv_image = numpy.array(background)
    background = open_cv_image.astype('float32')
    background = background / 255
    background = background.reshape((1,) + background.shape)
    predictions = model.predict_classes(background)

    
    print predictions
    key = (key for key, value in classes.items() if value == predictions[0]).next()
    return key


def nothing(x):
    pass



cv2.namedWindow('Camera Output')
cv2.namedWindow('Hand')
cv2.namedWindow('HandTrain')


cv2.createTrackbar('B for min', 'Camera Output', 0, 255, nothing)
cv2.createTrackbar('G for min', 'Camera Output', 0, 255, nothing)
cv2.createTrackbar('R for min', 'Camera Output', 0, 255, nothing)
cv2.createTrackbar('B for max', 'Camera Output', 0, 255, nothing)
cv2.createTrackbar('G for max', 'Camera Output', 0, 255, nothing)
cv2.createTrackbar('R for max', 'Camera Output', 0, 255, nothing)

cv2.setTrackbarPos('B for min', 'Camera Output', 0)
cv2.setTrackbarPos('G for min', 'Camera Output', 130)
cv2.setTrackbarPos('R for min', 'Camera Output', 103)
cv2.setTrackbarPos('B for max', 'Camera Output', 255)
cv2.setTrackbarPos('G for max', 'Camera Output', 182)
cv2.setTrackbarPos('R for max', 'Camera Output', 130)


videoFrame = cv2.VideoCapture(0)

keyPressed = -1  
palm_cascade = cv2.CascadeClassifier(' classificateur en cascade')


x_crop_prev, y_crop_prev, w_crop_prev, h_crop_prev = 0, 0, 0, 0


_, prevHandImage = videoFrame.read()


prevcnt = numpy.array([], dtype=numpy.int32)


gestureStatic = 0
gestureDetected = 0

while keyPressed < 0:  
   
    min_YCrCb = numpy.array([cv2.getTrackbarPos('B for min', 'Camera Output'),
                             cv2.getTrackbarPos('G for min', 'Camera Output'),
                             cv2.getTrackbarPos('R for min', 'Camera Output')], numpy.uint8)
    max_YCrCb = numpy.array([cv2.getTrackbarPos('B for max', 'Camera Output'),
                             cv2.getTrackbarPos('G for max', 'Camera Output'),
                             cv2.getTrackbarPos('R for max', 'Camera Output')], numpy.uint8)

  
    readSucsess, sourceImage = videoFrame.read()

   
    imageYCrCb = cv2.cvtColor(sourceImage, cv2.COLOR_BGR2YCR_CB)
    imageYCrCb = cv2.GaussianBlur(imageYCrCb, (5, 5), 0)

    skinRegion = cv2.inRange(imageYCrCb, min_YCrCb, max_YCrCb)

    _, contours, hierarchy = cv2.findContours(skinRegion, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    cnt = contours[0]
    ret = cv2.matchShapes(cnt, prevcnt, 2, 0.0)
    prevcnt = contours[0]

  
    stencil = numpy.zeros(sourceImage.shape).astype(sourceImage.dtype)
    color = [255, 255, 255]
    cv2.fillPoly(stencil, [cnt], color)
    handTrainImage = cv2.bitwise_and(sourceImage, stencil)

   
    if (ret > 0.70):
        gestureStatic = 0
    else:
        gestureStatic += 1


    x_crop, y_crop, w_crop, h_crop = cv2.boundingRect(cnt)

  
    cv2.rectangle(sourceImage, (x_crop, y_crop), (x_crop + w_crop, y_crop + h_crop), (0, 255, 0), 2)

  
    if (abs(x_crop - x_crop_prev) > 50 or abs(y_crop - y_crop_prev) > 50 or
                abs(w_crop - w_crop_prev) > 50 or abs(h_crop - h_crop_prev) > 50):
        x_crop_prev = x_crop
        y_crop_prev = y_crop
        h_crop_prev = h_crop
        w_crop_prev = w_crop

    handImage = sourceImage.copy()[max(0, y_crop_prev - 50):y_crop_prev + h_crop_prev + 50,
                max(0, x_crop_prev - 50):x_crop_prev + w_crop_prev + 50]

   
    handTrainImage = handTrainImage[max(0, y_crop_prev - 15):y_crop_prev + h_crop_prev + 15,
                     max(0, x_crop_prev - 15):x_crop_prev + w_crop_prev + 15]

 
    if gestureStatic == 10:
        gestureDetected = 10;
        print("Gesture Detected")
        letterDetected = identifyGesture(handTrainImage) 

    if gestureDetected > 0:
        if (letterDetected != None):
            cv2.putText(sourceImage, letterDetected, (10, 400), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 2)
        gestureDetected -= 1

   
    gray = cv2.cvtColor(handImage, cv2.COLOR_BGR2HSV)
    palm = palm_cascade.detectMultiScale(gray)
    for (x, y, w, h) in palm:
        cv2.rectangle(sourceImage, (x, y), (x + w, y + h), (255, 0, 0), 2)
      
        roi_color = sourceImage[y:y + h, x:x + w]

    hull = cv2.convexHull(cnt, returnPoints=False)
    defects = cv2.convexityDefects(cnt, hull)

    
    count_defects = 0
    for i in range(defects.shape[0]):
        s, e, f, d = defects[i, 0]
        start = tuple(cnt[s][0])
        end = tuple(cnt[e][0])
        far = tuple(cnt[f][0])
        if count_defects == 0:
            center_of_palm = far
        a = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
        b = math.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
        c = math.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
        angle = math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c)) * 57
        if angle <= 90:
            count_defects += 1
            if count_defects < 5:
               
                center_of_palm = (far[0] + center_of_palm[0]) / 2, (far[1] + center_of_palm[1]) / 2
        cv2.line(sourceImage, start, end, [0, 255, 0], 2)
   


    cv2.drawContours(sourceImage, contours, 0, (0, 255, 0), 1)


    cv2.imshow('Camera Output', sourceImage)
    cv2.imshow('Hand', handImage)
    cv2.imshow('HandTrain', handTrainImage)

   
    keyPressed = cv2.waitKey(30)  
cv2.destroyWindow('Camera Output')
videoFrame.release()
