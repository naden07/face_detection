import cv2

frontal_face = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

video_capture = cv2.VideoCapture(0)

while True:
    c_rec, d_image = video_capture.read()
    e = cv2.cvtColor(d_image, cv2.COLOR_BGR2GRAY)
    f = frontal_face.detectMultiScale(e, 1.3,6)

    for (x1,y1, w1, h1) in f:
        cv2.rectangle(d_image, (x1, y1), (x1+w1, y1+h1), (255,0,0),5)

    cv2.imshow('img', d_image)
    h = cv2.waitKey(40) & 0xFF
    if h == 40:
        break

video_capture.release()
cv2.destroyAllWindows()