from sklearn.neighbors import KNeighborsClassifier
import cv2
import pickle
import numpy as np
import os
import time
from datetime import datetime
import csv
from win32com.client import Dispatch

def speak(str1):
    try:
        speak_engine = Dispatch(("SAPI.SpVoice"))
        speak_engine.Speak(str1)
    except:
        pass

# 1. إعداد الكاميرا
video = cv2.VideoCapture('http://192.168.1.2:81/stream') 
video.set(cv2.CAP_PROP_BUFFERSIZE, 1)

facedetect = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')

with open('data/names.pkl', 'rb') as w:
    LABELS = pickle.load(w)
with open('data/faces.pkl', 'rb') as f:
    FACES = pickle.load(f)

print('Shape of Faces matrix --> ', FACES.shape)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(FACES, LABELS)

COL_NAMES = ['NAME', 'TIME']

# التأكد من وجود الفولدر الرئيسي، لو مش موجود نكريته
if not os.path.exists("Attendance"):
    os.makedirs("Attendance")

# المتغيرات الخاصة بالوقت وتفادي التكرار
last_recorded_time = {} 
COOLDOWN_SECONDS = 3600 # ساعة كاملة (3600 ثانية)

while True:
    ret, frame = video.read()

    if not ret or frame is None:
        print("Waiting for camera stream...")
        continue

    frame = cv2.resize(frame, (640, 480))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        crop_img = frame[y:y+h, x:x+w, :]
        resized_img = cv2.resize(crop_img, (50, 50)).flatten().reshape(1, -1)
        output = knn.predict(resized_img)
        
        name_detected = str(output[0])
        ts = time.time()
        date = datetime.fromtimestamp(ts).strftime("%d-%m-%Y")
        timestamp = datetime.fromtimestamp(ts).strftime("%H:%M-%S")
        
        # رسم المربعات
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.rectangle(frame, (x, y-40), (x+w, y), (0, 255, 0), -1)
        cv2.putText(frame, name_detected, (x, y-15), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1)
        
        # --- منطق الحفظ التلقائي وإنشاء الملف ---
        should_record = False
        
        if name_detected not in last_recorded_time:
            should_record = True
        else:
            time_diff = ts - last_recorded_time[name_detected]
            if time_diff > COOLDOWN_SECONDS:
                should_record = True
        
        if should_record:
            speak(f"Welcome {name_detected}")
            
            attendance = [name_detected, str(timestamp)]
            file_path = "Attendance/Attendance_" + date + ".csv"
            
            # بنشوف الملف موجود ولا لأ قبل ما نفتح عشان نقرر هنكتب العناوين ولا لأ
            file_exists = os.path.isfile(file_path)

            # وضع 'a' بيفتح الملف للكتابة، ولو مش موجود بيكريته هو
            with open(file_path, "a", newline='') as csvfile: 
                writer = csv.writer(csvfile)
                
                # لو الملف لسه مخلوق جديد (يعني مكنش موجود)، نكتب العناوين الأول
                if not file_exists:
                    writer.writerow(COL_NAMES)
                    
                writer.writerow(attendance)
            
            last_recorded_time[name_detected] = ts
            print(f"Recorded: {name_detected} at {timestamp}")

    cv2.imshow("Face Recognition", frame)

    k = cv2.waitKey(1)
    if k == ord('q'):
        break

video.release()
cv2.destroyAllWindows()