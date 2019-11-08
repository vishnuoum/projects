import cv2

# Load the cascade
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
face_cascade1=cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
face_cascade2=cv2.CascadeClassifier('haarcascade_frontalface_alt_tree.xml')
face_cascade3=cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')
eye_cascade=cv2.CascadeClassifier('haarcascade_eye.xml')

def show_webcam(mirror):

    #capture camera image
    cam = cv2.VideoCapture(0)
    
    while True:

        #reads camera image and convert to numpy array
        ret_val, img = cam.read()

        #convert to gray
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        #face detection
        faces = face_cascade1.detectMultiScale(gray, 1.1, 4)

        #eye detection
        #eyes=eye_cascade.detectMultiScale(gray,1.1,5)

        
        # Draw rectangle around the faces
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

        #draw rectanlge around eyes
        #for (x, y, w, h) in eyes:
            #cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        img = cv2.flip(img, 1)

        
        cv2.imshow('image', img)

        
        if cv2.waitKey(1) == 27:
            cam.release()
            break  # esc to quit
        
    cv2.destroyAllWindows()


def main():
    show_webcam(True)


main()
