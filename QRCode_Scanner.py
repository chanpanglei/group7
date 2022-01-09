import cv2
import webbrowser

cap = cv2.VideoCapture(0)
detector = cv2.QRCodeDetector()
while True:
    _, img = cap.read()
    data, one, _ = detector.detectAndDecode(img)  # img is from external
    if data:
        a = data  # data may be website or s string of Numbers(RAW), and put it in the box 'a'
        break
    cv2.imshow('QRCode Scanner', img)  # new window is called 'qrcodescanner app'
    if cv2.waitKey(1) == ord('q'):
        break
b = webbrowser.open(str(a))
cap.release(a)
cv2.destroyWindow()
