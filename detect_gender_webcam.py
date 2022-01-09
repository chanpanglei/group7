#!/usr/bin/python3
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import os
import cvlib as cv

# load the overlay image. size should be smaller than video frame size
imgM = cv2.imread("Database/stars/M/01.png")
imgF = cv2.imread("Database/stars/F/02.png")

# Get Image dimensions
imgM_height, imgM_width, _ = imgM.shape
imgF_height, imgF_width, _ = imgF.shape

x = 10
y = 10

# load model
model = load_model('gender_detection.model')

# open webcam
webcam = cv2.VideoCapture(0)
fps = webcam.get(cv2.CAP_PROP_FPS)    # Get the frame rate of the video
timeF = int(fps)     # Video frame count interval frequency
n = 1  # counter
i = 3
classes = ['man', 'woman']

# loop through frames
while webcam.isOpened():
    # read frame from webcam 
    status, frame = webcam.read()

    # apply face detection
    face, confidence = cv.detect_face(frame)
    if (n % timeF == 0):
        i -= 1
    n = n + 1
    # loop through detected faces
    for idx, f in enumerate(face):

        # get corner points of face rectangle        
        (startX, startY) = f[0], f[1]
        (endX, endY) = f[2], f[3]

        # draw rectangle over face
        cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)

        # crop the detected face region
        face_crop = np.copy(frame[startY:endY, startX:endX])

        if (face_crop.shape[0]) < 10 or (face_crop.shape[1]) < 10:
            continue

        # preprocessing for gender detection model
        face_crop = cv2.resize(face_crop, (96, 96))
        face_crop = face_crop.astype("float") / 255.0
        face_crop = img_to_array(face_crop)
        face_crop = np.expand_dims(face_crop, axis=0)

        # apply gender detection on face
        conf = model.predict(face_crop)[0]  # model.predict return a 2D matrix, ex: [[9.9993384e-01 7.4850512e-05]]

        # get label with max accuracy
        idx = np.argmax(conf)
        # print(idx)
        # add image to frame
        if idx == 0:
            imgF = cv2.imread("Database/stars/F/01.png")
            frame[y:y + imgF_height, x:x + imgF_width] = imgF
        else:
            imgM = cv2.imread("Database/stars/M/01.png")
            frame[y:y + imgM_height, x:x + imgM_width] = imgM

        label = classes[idx]

        label = "{}: {:.2f}%".format(label, conf[idx] * 100)

        Y = startY - 10 if startY - 10 > 10 else startY + 10
        # write label and confidence above face rectangle
        cv2.putText(frame, label, (startX, Y),  cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 0), 2)
    if i == 0:
        cv2.imwrite("D:/photobase/" + str(i) + '.jpg', frame)
        break
    else:
        cv2.putText(frame, str(i), (500, 50), cv2.FONT_HERSHEY_SIMPLEX,
                    2, (0, 0, 0), 3)
    # display output
    cv2.imshow("gender detection", frame)

    # press "Q" to stop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# release resources
webcam.release()
cv2.destroyAllWindows()
