import  sys
import cv2
import numpy as np
from pyzbar.pyzbar import decode

cap1 = cv2.VideoCapture(1)
width = 640
high = 480
cap1.set(3, width)  # 3 is width for 640 pixels
cap1.set(4, high)  # 4 is high for 480 pixels

width = 640
high = 480
half_w = int(width * 0.5)
half_h = int(high * 0.5)

camera = True
with open('database/product_List') as f:
    P_List = f.read().splitlines()

while camera:
    success, barcode_frame = cap1.read()
    #success2, frame2 = cap.read()
    cv2.rectangle(barcode_frame, (half_w-80, half_h-80), (half_w+80, half_h+80), (255, 255, 255), 1)
    green_color = (136, 177, 18)  # BGR
    points1 = np.array([[half_w-40, half_h-80], [half_w-80, half_h-80], [half_w-80, half_h-40]], np.int32)
    cv2.polylines(barcode_frame, pts=[points1], isClosed=False, color=green_color, thickness=9)  # the upper left position
    points2 = np.array([[half_w+40, half_h-80], [half_w+80, half_h-80], [half_w+80, half_h-40]], np.int32)
    cv2.polylines(barcode_frame, pts=[points2], isClosed=False, color=green_color, thickness=9)  # the upper right position
    points3 = np.array([[half_w-80, half_h+40], [half_w-80, half_h+80], [half_w-40, half_h+80]], np.int32)
    cv2.polylines(barcode_frame, pts=[points3], isClosed=False, color=green_color, thickness=9)  # the lower left position
    points4 = np.array([[half_w+80, half_h+40], [half_w+80, half_h+80], [half_w+40, half_h+80]], np.int32)
    cv2.polylines(barcode_frame, pts=[points4], isClosed=False, color=green_color, thickness=9)  # the lower right position
    #cv2.putText(frame, 'Scan Area', (half_w-80, int(half_h*0.5)), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
    for code in decode(barcode_frame):
        print(code.type)
        P_code = code.data.decode('utf-8')
        print(P_code)
        if P_code in P_List:
            myOutput = 'Recognized'
            myColor = (136, 177, 18)
            if cv2.waitKey(1) == ord('a'):
                cap1.release()
                camera = False
                break

        else:
            myOutput = 'Unrecognized'
            myColor = (0, 0, 255)
            with open('database/unknown_list', 'a') as ul:
                ul.write(str(P_code) + '\n')

        cv2.putText(barcode_frame, myOutput, (half_w-100, half_h), cv2.FONT_HERSHEY_COMPLEX, 1, myColor, 2)

    cv2.imshow('Barcode Scanner', barcode_frame)
    if cv2.waitKey(1) == ord('q'):
        cap1.release()
        cv2.destroyAllWindows()
        sys.exit()
cap1.release()
cv2.destroyAllWindows()
