from picamera.array import PiRGBArray
from picamera import PiCamera
import cv2
import os
import zipfile
import shutil

# Tạo thư mục để lưu hình ảnh
face_id = input("\n Nhập mssv: ")
folder = f'datasets/{face_id}'
if not os.path.exists(folder):
    os.makedirs(folder)

# Khởi tạo PiCamera
camera = PiCamera()
camera.resolution = (480, 320)
camera.framerate = 30
rawCapture = PiRGBArray(camera, size=(480, 320))

# Load file cascade
face_cascade = cv2.CascadeClassifier('/home/admin/Desktop/Face_recognition/haarcascade_frontalface_default.xml')

print("\n Nhìn vào camera chờ chụp hình...")
count = 100

# Chụp hình
for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
    image = frame.array
    image = cv2.flip(image, -1)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(100, 100), flags=cv2.CASCADE_SCALE_IMAGE)
    for (x, y, w, h) in faces:
        cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,0),2)
    cv2.imshow("Frame", image)

    key = cv2.waitKey(1)
    if key == ord("q"):
        break
    elif key == 32:
        img_path = f'{folder}/Image {count}.jpg'
        cv2.imwrite(img_path, gray[y:y+h, x:x+w])
        count += 1
    if count > 120:
        print("\n Đã chụp hình xong...")
        break 
        
    rawCapture.truncate(0)
cv2.destroyAllWindows()







