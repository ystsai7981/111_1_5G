import cv2

def detectAndDisplay(frame):
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_gray = cv2.equalizeHist(frame_gray)

    #-- Detect faces
    faces = face_cascade.detectMultiScale(frame_gray)
    for (x,y,w,h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
    cv2.imshow('Capture - Face detection', frame)

if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    cascadepath = 'haarcascades/haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(cascadepath)
    while True:
        ret, frame = cap.read()
        if frame is None:
            break
        detectAndDisplay(frame)
        if cv2.waitKey(1) == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()