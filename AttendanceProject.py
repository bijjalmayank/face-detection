import requests
import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime, timedelta
from geopy.distance import distance

path = 'ImagesAttendance'
images = []
classNames = []
myList = os.listdir(path)
print(myList)

for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])

print(classNames)

# Caching mechanism for location
cached_location = None
last_location_time = datetime.min

def get_current_location():
    global cached_location, last_location_time
    if datetime.now() - last_location_time > timedelta(seconds=15):  # Update location every 15 seconds
        try:
            response = requests.get('http://www.geoplugin.net/json.gp')
            response.raise_for_status()
            data = response.json()
            if 'geoplugin_latitude' in data and 'geoplugin_longitude' in data:
                cached_location = float(data['geoplugin_latitude']), float(data['geoplugin_longitude'])
                last_location_time = datetime.now()
                print(f"Updated location: {cached_location}")
            else:
                print("Location data not found in response:", data)
                cached_location = None, None
        except requests.exceptions.RequestException as e:
            print(f"Error: {e}")
            cached_location = None, None
    return cached_location

def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

def is_within_area(lat, lon, center_lat, center_lon, radius):
    device_location = (lat, lon)
    center_location = (center_lat, center_lon)
    distance_km = distance(device_location, center_location).km
    print(f"Distance from center: {distance_km} km")
    return distance_km <= radius

def markAttendance(name, center_lat, center_lon, radius):
    current_lat, current_lon = get_current_location()
    print(f"Device location: ({current_lat}, {current_lon})")
    print(f"Center location: ({center_lat}, {center_lon})")
    if current_lat is None or current_lon is None or not is_within_area(current_lat, current_lon, center_lat, center_lon, radius):
        print(f"Location mismatch or unable to determine location: ({current_lat}, {current_lon}) not within radius")
        return

    print(f"Marking attendance for {name} at location ({current_lat}, {current_lon})")

    with open('Attendance.csv', 'a+') as f:
        f.seek(0)
        myDataList = f.readlines()

        now = datetime.now()
        currentDate = now.strftime('%Y-%m-%d')
        currentTime = now.strftime('%H:%M:%S')

        if not any(f'{name},{currentDate}' in entry for entry in myDataList):
            dtString = now.strftime('%Y-%m-%d %H:%M:%S')
            f.write(f'\n{name},{dtString}')
            print(f'Attendance marked for {name} at {dtString}')

# Replace with the latitude, longitude, and radius of the area you want to allow
center_lat = 12.9094  # Example: Latitude of Bengaluru city center
center_lon = 77.5668  # Example: Longitude of Bengaluru city center
radius = 10  # Radius in kilometers

encodeListKnown = findEncodings(images)
print('Encoding Complete')

cap = cv2.VideoCapture(0)
attendance_marked = set()
display_faces = {}

while True:
    success, img = cap.read()
    if not success:
        break

    imgS = cv2.resize(img, (0, 0), None, 0.5, 0.5)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    facesCurFrame = face_recognition.face_locations(imgS)
    encodeCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

    for encodeFace, faceLoc in zip(encodeCurFrame, facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 * 2, x2 * 2, y2 * 2, x1 * 2
            display_faces[name] = ((x1, y1, x2, y2), datetime.now())

            if name not in attendance_marked:
                markAttendance(name, center_lat, center_lon, radius)
                markAttendance(name, center_lat, center_lon, radius)
                attendance_marked.add(name)

    current_time = datetime.now()
    for name, ((x1, y1, x2, y2), timestamp) in list(display_faces.items()):
        if (current_time - timestamp).seconds > 1:  # Remove face display after 1 second
            del display_faces[name]
        else:
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

    imgBGR = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    cv2.imshow('Webcam', imgBGR)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
