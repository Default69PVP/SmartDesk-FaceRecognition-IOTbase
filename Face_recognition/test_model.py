from picamera.array import PiRGBArray
from picamera import PiCamera
import cv2
import os
import subprocess
import time
import socket
from datetime import datetime
import firebase_admin
from firebase_admin import credentials
from firebase_admin import db
# Khởi tạo camera
camera = PiCamera()
camera.resolution = (480, 368)
camera.framerate = 30
rawCapture = PiRGBArray(camera, size=(480, 368))
# Koi tao firebase
cred = credentials.Certificate('/home/admin/Desktop/Face_recognition/credentialkey.json')
firebase_admin.initialize_app(cred, {"databaseURL": "https://database-cbe89-default-rtdb.asia-southeast1.firebasedatabase.app/"})
ref = db.reference("/danh_sach_lop/ce001/diem_danh_theo_ngay/")

#Local Binary Patterns Histograms
recognizer = cv2.face.LBPHFaceRecognizer_create()

#Load file train
recognizer.read('/home/admin/Desktop/Face_recognition/Trainer.yml')

#Load file cascade
face_cascade = cv2.CascadeClassifier('/home/admin/Desktop/Face_recognition/haarcascade_frontalface_default.xml')
font = cv2.FONT_HERSHEY_SIMPLEX


for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
    image = frame.array
    image = cv2.flip(image,-1)
    
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor = 1.2, minNeighbors = 5, minSize = (100, 100), flags = cv2.CASCADE_SCALE_IMAGE)
    
    # Nhận diện
    for (x,y,w,h) in faces:
        roi_gray = gray[y:y + h, x:x + w]
        cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,0),2)
        id, confidence = recognizer.predict(gray[y:y+h,x:x+w])
        id = str(id)
        folder = f'Newdataset/{id}'
        if not os.path.exists(folder):
            os.makedirs(folder)
#       confidence = 100 - confidence
        if (45< confidence < 55):
            d = datetime.now()
            dformat = '%Y-%m-%d'
            tformat = '%H:%M:%S'
            date = d.strftime(dformat)
            time1 = d.strftime(tformat)
            
            confidence = "{0}%".format(round(confidence))
            count = len(os.listdir(f'Newdataset/{id}'))
            img_path = f'{folder}/Image{count}.jpg'
            cv2.imwrite(img_path, roi_gray)
            print(id, confidence)
            
            count = 0
            length = 0
            
            for user in ref.get():
                length = length + 1
            for user in ref.get():
                if(user["mssv"] == id):
                    if(user["date"] == date):
                        ref.child(str(count)).update(
                        {
                            "status": "1"
                        })
                        break
                if(count == length - 1):
                    ref.child(str(length)).update(
                        {
                          "mssv": id,
                          "status": "1",
                          "date": date
                        })
                count = count + 1
            
            print(id)
            print(time1)

            
            #Restart
            time.sleep(3)
            camera.close() 
            #rawCapture.truncate(0)
            cv2.destroyAllWindows()
            
            subprocess.call(['python','/home/admin/Desktop/Face_recognition/test_model.py'])
        else:
            id = "unknown"
            confidence = "0%"
            cv2.putText(image, str(id), (x+5,y-5), font, 1, (255,255,255), 2)
            print(id, confidence)
    #cv2.imshow("Frame", image)
  
        
    if cv2.waitKey(1) & 0xff == ord("q"):
        camera.close()
        exit()
   
    rawCapture.truncate(0)
    