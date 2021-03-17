# Tamim Shaban
# Note: Please Enter a valid email address and password

import cv2
import numpy as np
from keras.models import load_model
import imghdr
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage

face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
classifier = load_model('./Emotion_Detection.h5')

cap = cv2.VideoCapture(0)

faceCount = 0

server = smtplib.SMTP('smtp.gmail.com: 587')
password = "your_password"
msg_from = "your_address"
msg_to = "to_address"

while True:
    _, frame = cap.read()
    frame = cv2.flip(frame, 1)
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(frame_gray, 1.3, 5)
    if len(faces) > 0:
        faceCount += 1
        cv2.imwrite(str(faceCount) + ".jpeg", frame)

        msg = MIMEMultipart()
        msg['From'] = msg_from
        msg['To'] = msg_to
        msg['Subject'] = 'Assignment 9: Face Detector'
        body = "Face Detected\nImage attached below:"
        attachment = str(faceCount) + ".jpeg"
        msgText = MIMEText('<b>%s</b><br><img src="cid:%s"><br>' % (body, attachment), 'html')
        msg.attach(msgText)
        fp = open(attachment, 'rb')
        img = MIMEImage(fp.read())
        fp.close()
        img.add_header('Content-ID', '<{}>'.format(attachment))
        msg.attach(img)

        server.starttls()
        server.login(msg['From'], password)
        server.sendmail(msg['From'], msg['To'], msg.as_string())
        server.quit()
    
    if cv2.waitKey(1) & 0XFF == ord('q'):
        break


        
