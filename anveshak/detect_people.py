import cv2
import numpy as np
from darkflow.net.build import TFNet

video_path = r"C:\Users\anves\Documents\UML\IOT\IOT_group_project\people_walking.mp4"

options = {"model": "yolov2-tiny.cfg", "load":"yolov2-tiny.weights", "threshold":0.1}
tfnet = TFNet(options)

cap = cv2.VideoCapture(video_path)

if not cap.isOpened:
    print("Error opening video capture")
    exit(0)

while(True):
    ret, frame = cap.read()
    if frame is None:
        pass

    # frame = detectPeople(frame)

    result = tfnet.return_predict(np.array(frame))
    print(result)
    cv2.imshow("output", frame)

    if cv2.waitKey(1) & 0xFF == ord('q' or 'Q'):
        break