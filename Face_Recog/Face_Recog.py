''''
Real Time Face Recogition
	==> Each face stored on dataset/ dir, should have a unique numeric integer ID as 1, 2, 3, etc                       
	==> LBPH computed model (trained faces) should be on trainer/ dir
Based on original code by Anirban Kar: https://github.com/thecodacus/Face-Recognition    
Developed by Marcelo Rovai - MJRoBot.org @ 21Feb18  
'''

from datetime import datetime
import cv2
import numpy as np
import os
import subprocess
from socket import *
import json


#---------------------------------Face_Recognition----------------------------------
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer/trainer.yml')
cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath);

font = cv2.FONT_HERSHEY_SIMPLEX

#iniciate id counter
id = 0

# names related to ids: example ==> Marcelo: id=1,  etc
names = ['None', 'Kelvin', 'Mattew']

# Initialize and start realtime video capture
cam = cv2.VideoCapture(0)
cam.set(3, 480) # set video widht
cam.set(4, 320) # set video height

# Define min window size to be recognized as a face
minW = 0.1*cam.get(3)
minH = 0.1*cam.get(4)

cnt = 0
subprocess.call('amixer -c 1 cset numid=2 off', shell=True)

timeList = {}
timeList['AT'] = {}
timeList['LT'] = {}

#------------------------------------------------------------------------------------

#--------------------------------------Socket----------------------------------------
HOST='210.115.230.129'

c = socket(AF_INET, SOCK_STREAM)
print('connecting....')
c.connect((HOST,65000))
print('ok')

#------------------------------------------------------------------------------------
while True:
    ret, img =cam.read()

    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale( 
        gray,
        scaleFactor = 1.2,
        minNeighbors = 5,
        minSize = (int(minW), int(minH)),
       )

    cnt = cnt + 1
    if cnt == 180:
        subprocess.call('amixer -c 1 cset numid=2 off', shell=True)
        cnt = 0



    for(x,y,w,h) in faces:

        cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)

        id, confidence = recognizer.predict(gray[y:y+h,x:x+w])

        # Check if confidence is less them 100 ==> "0" is perfect match 
        if(confidence <= 50):
            id = names[id]
            confidence = "  {0}%".format(round(100 - confidence))
            timestamp = str(datetime.now())[:-7]
            asdf = {'name': 'kelvin','access_date': timestamp}
            data = json.dumps(asdf)
            c.send(str.encode(data))
            subprocess.call('amixer -c 1 cset numid=2 on', shell=True)

        else:
            id = "unknown"
            confidence = "  {0}%".format(round(100 - confidence))
        
        cv2.putText(img, str(id), (x+5,y-5), font, 1, (255,255,255), 2)
        cv2.putText(img, str(confidence), (x+5,y+h-5), font, 1, (255,255,0), 1)  



    cv2.imshow('camera',img) 


    k = cv2.waitKey(10) & 0xff # Press 'ESC' for exiting video
    if k == 27:
        break

# Do a bit of cleanup
print("\n [INFO] Exiting Program and cleanup stuff")
c.close()
cam.release()
cv2.destroyAllWindows()
