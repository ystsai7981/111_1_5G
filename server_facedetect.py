import cv2
import sys

def face_detect(image):

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor = 1.2,
        minNeighbors = 5,
        minSize = (30,30)
        )

    num_faces = len(faces)
    return num_faces,faces


if __name__ == "__main__":
    ##沒有gui
    cascasdepath = '/haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + cascasdepath)

    video_capture = cv2.VideoCapture(0)
    num_faces = 0

    while True:
        ret, image = video_capture.read()

        if not ret:
            break

        num_faces,faces = face_detect(image)
        print("The number of faces found = ", len(faces))
    video_capture.release()
    cv2.destroyAllWindows()