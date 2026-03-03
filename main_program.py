import cv2

frontal_face = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

video_capture = cv2.VideoCapture(0)

while True:
    capture_success, frame = video_capture.read()
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detected_faces = frontal_face.detectMultiScale(gray_frame, 1.3, 6)

    for (x1,y1, w1, h1) in detected_faces:
        cv2.rectangle(frame, (x1, y1), (x1+w1, y1+h1), (255,0,0),5)

    cv2.imshow('img', frame)
    pressed_key = cv2.waitKey(40) & 0xFF
    if  pressed_key== 40:
        break

video_capture.release()
cv2.destroyAllWindows()