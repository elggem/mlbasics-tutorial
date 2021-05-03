# Live "six basic emotion" recognition example using Webcam and OpenCV
#
# sources
# https://github.com/shantnu/Webcam-Face-Detect
# https://github.com/nhduong/fer2013/blob/master/fer2013.ipynb

import cv2
import sys
import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l1, l2

# Face detection classifier model loading
cascPath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)

# Turn on Webcam
video_capture = cv2.VideoCapture(0)


# Load model for FER
n_classes = 7

model = Sequential()
model.add(Conv2D(64, (3, 3), activation='relu', padding='same', input_shape = (48, 48, 1)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(256, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(1024, activation='relu', kernel_regularizer=l2(0.001)))
model.add(Dropout(0.5))
model.add(Dense(n_classes, activation='softmax'))
opt = Adam(lr=0.0001, decay=10e-6)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

# Load model weights from pretrained model
model.load_weights('pretrained/fer2013_weights.h5')

# Emotion indices according to dataset labels
EMOTIONS = {
    0: "anger",
    1: "disgust",
    2: "fear",
    3: "happy",
    4: "sad",
    5: "surprise",
    6: "neutral"
}

# Colors
cnn_clr = (0, 0, 255)
txt_clr = (255, 255, 255)

# Loop until we quit
while True:
    # Capture video frame
    ret, frame = video_capture.read()
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    # For each detected face...
    for (x, y, w, h) in faces:
        # draw rectangle around face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # perform classification
        roi = gray[y:y+h,x:x+w]
        roi = cv2.resize(roi, (48,48))
        cv2.imshow('ROI', roi)
        pred_cls = model.predict(roi.reshape(1, 48, 48, 1))[0]

        # display
        emo_dict = {}
        for i, emotype in enumerate(EMOTIONS):
            emo_dict[EMOTIONS[emotype]] = pred_cls[i]
        p = 0
        for emotype, emo in sorted(emo_dict.items()):
            cv2.rectangle(frame, (x, y + h - 7 * 20 + (p * 20)), (x + (int(emo * 80)), y + h - 7 * 20 + (p * 20) + 20), txt_clr, -1)
            cv2.putText(frame, emotype, (x, 15 + y + h - 7 * 20 + (p * 20)), cv2.FONT_HERSHEY_DUPLEX, 0.55, cnn_clr)
            p += 1

    # Display the resulting frame
    cv2.imshow('Video', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()