# Library for Azure Kinect DK
import sys
import pykinect_azure as pykinect
import cv2
import time
###############################################################
# Library for face detection
# import cv2
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import os
import cvlib as cv
###############################################################
# Library for barcode
# import cv2
# import numpy as np
from pyzbar.pyzbar import decode
###############################################################
sys.path.insert(1, '../')
##############################################
# load model
model = load_model('gender_detection.model')
classes = ['man', 'woman']
##############################################
with open('database/product_List') as f:
    P_List = f.read().splitlines()
##############################################
camera = True
##############################################
if __name__ == "__main__":

    # Initialize the library, if the library is not found, add the library path as argument
    pykinect.initialize_libraries()

    # Modify camera configuration
    device_config = pykinect.default_configuration
    device_config.color_format = pykinect.K4A_IMAGE_FORMAT_COLOR_BGRA32
    device_config.color_resolution = pykinect.K4A_COLOR_RESOLUTION_720P
    device_config.depth_mode = pykinect.K4A_DEPTH_MODE_WFOV_2X2BINNED
    # print(device_config)

    # Start device
    device = pykinect.start_device(config=device_config)
    cv2.namedWindow('Transformed Color Image', cv2.WINDOW_NORMAL)

    while camera:
        # Get capture
        capture = device.update()

        # Get the color image from the capture
        ret, color_image = capture.get_transformed_color_image()

        if not ret:
            continue

        # Get the colored depth
        ret, depth_image = capture.get_colored_depth_image()

        # Combine both images
        combined_image = cv2.addWeighted(color_image[:, :, :3], 0.7, depth_image, 0.3, 0)

        # apply face detection
        face, confidence = cv.detect_face(combined_image)
        # loop through detected faces
        for idx, f in enumerate(face):
            # get corner points of face rectangle
            (startX, startY) = f[0], f[1]
            (endX, endY) = f[2], f[3]

            # draw rectangle over face
            cv2.rectangle(combined_image, (startX, startY), (endX, endY), (0, 255, 0), 2)

            # crop the detected face region
            face_crop = np.copy(combined_image[startY:endY, startX:endX])

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
            label = classes[idx]
            label = "{}: {:.2f}%".format(label, conf[idx] * 100)
            Y = startY - 10 if startY - 10 > 10 else startY + 10

            # write label and confidence above face rectangle
            cv2.putText(combined_image, label, (startX, Y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 255, 0), 2)

            if label == "man" or "woman":
                print("!!!yes!!!")
                # Press a key to access
                if cv2.waitKey(1) == ord('a'):
                    capture.release()
                    cv2.destroyAllWindows()
                    camera = False
                    break

        # Overlay body segmentation on depth image
        cv2.imshow('Transformed Color Image', combined_image)
        # Press q key to stop
        if cv2.waitKey(1) == ord('q'):
            capture.release()
            cv2.destroyAllWindows()
            sys.exit()
    ####################################################################################################################
time.sleep(1)
confThreshold = 0.4
cap = cv2.VideoCapture(0)

# Create an empty list - classes[] and point the classesFile to 'coco80.names'
classesFile = 'coco80.names'
classes = []
# Load all classes in coco80.names into classes[]
with open(classesFile, 'r') as f:
    classes = f.read().splitlines()
    # print(classes)
    # print(len(classes))

    # Load the configuration and weights file
    # You need to download the weights and cfg files from https://pjreddie.com/darknet/yolo/
    net = cv2.dnn.readNetFromDarknet('yolov3-320.cfg', 'yolov3-320.weights')
    # Use OpenCV as backend and use CPU
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    while True:
        success, img = cap.read()
        height, width, ch = img.shape

        blob = cv2.dnn.blobFromImage(img, 1 / 255, (320, 320), (0, 0, 0), swapRB=True, crop=False)
        net.setInput(blob)

        layerNames = net.getLayerNames()
        print(layerNames)

        output_layers_names = net.getUnconnectedOutLayersNames()
        print(output_layers_names)

        LayerOutputs = net.forward(output_layers_names)
        print(len(LayerOutputs))
        # print(LayerOutputs[0].shape)
        # print(LayerOutputs[1].shape)
        # print(LayerOutputs[2].shape)
        # print(LayerOutputs[0][0])

        bboxes = []  # array for all bounding boxes of detected classes
        confidences = []  # array for all confidence values of matching detected classes
        class_ids = []  # array for all class IDs of matching detected classes

        for output in LayerOutputs:
            for detection in output:
                scores = detection[5:]  # omit the first 5 values
                class_id = np.argmax(
                    scores)  # find the highest score ID out of 80 values, has the highest confidence value
                confidence = scores[class_id]
                if confidence > confThreshold:
                    center_x = int(detection[0] * width)  # YOLO predicts centers of image
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    bboxes.append([x, y, w, h])
                    confidences.append((float(confidence)))
                    class_ids.append(class_id)
                    # cv2.rectangle(img, (x, y), (x + w, y + h), (255,0,0), 2)

        # print(len(bboxes))
        indexes = cv2.dnn.NMSBoxes(bboxes, confidences, confThreshold, 0.4)  # Non-maximum suppression
        # print(indexes)
        # print(indexes.flatten())

        font = cv2.FONT_HERSHEY_PLAIN
        colors = np.random.uniform(0, 255, size=(len(bboxes), 3))

        if len(indexes) > 0:
            for i in indexes.flatten():
                x, y, w, h = bboxes[i]
                label = str(classes[class_ids[i]])
                confidence = str(round(confidences[i], 2))
                color = colors[i]
                cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                cv2.putText(img, label + " " + confidence, (x, y + 20), font, 2, (255, 255, 255), 2)
                if label == "bottle":
                    print("!!!yes!!!")
                    if cv2.waitKey(1) == ord('a'):
                        cap.release()
                        camera = False
                        break

        cv2.imshow('Detection Frame', img)
        if cv2.waitKey(1) == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
########################################################################################################################
    cap.release()
    cv2.destroyAllWindows()
    time.sleep(1)
    # cap1: set up barcode frame
    cap1 = cv2.VideoCapture(0)
    width = 640
    high = 480
    half_w = int(width * 0.5)
    half_h = int(high * 0.5)
    cap1.set(3, width)  # 3 is width for 640 pixels
    cap1.set(4, high)  # 4 is high for 480 pixels
    with open('database/product_List') as f:
        P_List = f.read().splitlines()
        while True:
            success, barcode_frame = cap1.read()
            cv2.rectangle(barcode_frame, (half_w - 80, half_h - 80), (half_w + 80, half_h + 80), (255, 255, 255), 1)
            green_color = (136, 177, 18)  # BGR
            points1 = np.array([[half_w - 40, half_h - 80], [half_w - 80, half_h - 80], [half_w - 80, half_h - 40]],
                               np.int32)
            cv2.polylines(barcode_frame, pts=[points1], isClosed=False, color=green_color,
                          thickness=9)  # the upper left position
            points2 = np.array([[half_w + 40, half_h - 80], [half_w + 80, half_h - 80], [half_w + 80, half_h - 40]],
                               np.int32)
            cv2.polylines(barcode_frame, pts=[points2], isClosed=False, color=green_color,
                          thickness=9)  # the upper right position
            points3 = np.array([[half_w - 80, half_h + 40], [half_w - 80, half_h + 80], [half_w - 40, half_h + 80]],
                               np.int32)
            cv2.polylines(barcode_frame, pts=[points3], isClosed=False, color=green_color,
                          thickness=9)  # the lower left position
            points4 = np.array([[half_w + 80, half_h + 40], [half_w + 80, half_h + 80], [half_w + 40, half_h + 80]],
                               np.int32)
            cv2.polylines(barcode_frame, pts=[points4], isClosed=False, color=green_color,
                          thickness=9)  # the lower right position
            # cv2.putText(frame, 'Scan Area', (half_w-80, int(half_h*0.5)), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
            for code in decode(barcode_frame):
                print(code.type)
                P_code = code.data.decode('utf-8')
                print(P_code)
                if P_code in P_List:
                    myOutput = 'Recognized'
                    myColor = (136, 177, 18)

                else:
                    myOutput = 'Unrecognized'
                    myColor = (0, 0, 255)
                cv2.putText(barcode_frame, myOutput, (half_w - 100, half_h), cv2.FONT_HERSHEY_COMPLEX, 1, myColor, 2)

            cv2.imshow('Barcode Scanner', barcode_frame)
            if cv2.waitKey(1) == ord('q'):
                break
        cap1.release()