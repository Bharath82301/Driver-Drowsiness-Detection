import cv2
import os
from keras.models import load_model
import numpy as np
from pygame import mixer

mixer.init()

# Specify the full path to the 'alarm.wav' file
sound_file_path = r'D:\Projects\Drowsiness detection\alarm.wav'

if not os.path.exists(sound_file_path):
    print("Error: 'alarm.wav' not found at the specified path.")
    exit()

sound = mixer.Sound(sound_file_path)

face = cv2.CascadeClassifier('D:\Projects\Drowsiness detection\haar cascade files\haarcascade_frontalface_alt.xml')
leye = cv2.CascadeClassifier('D:\Projects\Drowsiness detection\haar cascade files\haarcascade_lefteye_2splits.xml')
reye = cv2.CascadeClassifier('D:\Projects\Drowsiness detection\haar cascade files\haarcascade_righteye_2splits.xml')

lbl = ['Close', 'Open']

model = load_model('D:\Projects\Drowsiness detection\models\cnnCat2.h5')
path = os.getcwd()
cap = cv2.VideoCapture(0)  # Initialize the video capture

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

font = cv2.FONT_HERSHEY_COMPLEX_SMALL
count = 0
score = 0
thicc = 2
rpred = [99]
lpred = [99]

while True:
    ret, frame = cap.read()

    if not ret:
        print("Error: Could not read frame.")
        break

    height, width = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face.detectMultiScale(gray, minNeighbors=5, scaleFactor=1.1, minSize=(25, 25))
    left_eye = leye.detectMultiScale(gray)
    right_eye = reye.detectMultiScale(gray)
    
    cv2.rectangle(frame, (0, height-50), (280, height), (0, 0, 0), thickness=cv2.FILLED)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (100, 100, 100), 1)

        # Extract eye regions
        roi_gray = gray[y:y+h, x:x+w]
        left_eye = leye.detectMultiScale(roi_gray)
        right_eye = reye.detectMultiScale(roi_gray)

        for (ex, ey, ew, eh) in right_eye:
            r_eye = frame[y+ey:y+ey+eh, x+ex:x+ex+ew]
            r_eye = cv2.cvtColor(r_eye, cv2.COLOR_BGR2GRAY)
            r_eye = cv2.resize(r_eye, (24, 24))
            r_eye = r_eye / 255
            r_eye = r_eye.reshape(24, 24, -1)
            r_eye = np.expand_dims(r_eye, axis=0)
            rpred = np.argmax(model.predict(r_eye), axis=-1)

            confidence = model.predict(r_eye)[0]
            predicted_class = np.argmax(confidence)

            #cv2.putText(frame, f"Class: {predicted_class} Confidence: {confidence[predicted_class]:.2f}", (10, height-20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
            
            if predicted_class == 0:  # Assuming 0 represents closed eyes
                cv2.putText(frame, "Closed", (10, height-20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
                score += 1
            else:
                cv2.putText(frame, "Open", (10, height-20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
                # Decrease the score by a smaller value when eyes are open
                score -= 0.5  # Adjust this value as per your requirement

    if score < 0:
        score = 0

    cv2.putText(frame, 'Score: ' + str(score), (150, height-20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)

    if score > 10:
        # person is feeling sleepy, so we beep the alarm
        cv2.imwrite(os.path.join(path, 'image.jpg'), frame)
        try:
            sound.play()
        except:
            pass

        if thicc < 16:
            thicc = thicc + 2
        else:
            thicc = thicc - 2
            if thicc < 2:
                thicc = 2

        cv2.rectangle(frame, (0, 0), (width, height), (0, 0, 255), thicc)

    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
