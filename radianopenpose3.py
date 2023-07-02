import cv2
import numpy as np
import time
import math

#허리파일 9, 14, 12- 다리 좌우 사이 각도 측정

from flask import Flask, Response, render_template

width = 600
height = 240

BODY_PARTS = { "Head": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
                "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
                "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "Chest": 14,
                "Background": 15 }

POSE_PAIRS = [ ["Head", "Neck"], ["Neck", "RShoulder"], ["RShoulder", "RElbow"],
                ["RElbow", "RWrist"], ["Neck", "LShoulder"], ["LShoulder", "LElbow"],
                ["LElbow", "LWrist"], ["Neck", "Chest"], ["Chest", "RHip"], ["RHip", "RKnee"],
                ["RKnee", "RAnkle"], ["Chest", "LHip"], ["LHip", "LKnee"], ["LKnee", "LAnkle"] ]
    
protoFile = "pose_deploy_linevec_faster_4_stages.prototxt"
weightsFile = "pose_iter_160000.caffemodel"

net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)

cap = cv2.VideoCapture(0)  # 웹캠사용

cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

app = Flask(__name__)


def generate_virtual_frame():

    ret, frame = cap.read()

    imageHeight, imageWidth, _ = frame.shape

    inpBlob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (imageWidth, imageHeight), (0, 0, 0), swapRB=False, crop=False)

    net.setInput(inpBlob)

    output = net.forward()

    H = output.shape[2]
    W = output.shape[3]

    points = []

    #center_x = int(imageWidth * point[0] / W)
    #center_y = int(imageHeight * point[1] / H)
    for i in range(0, 15):
        probMap = output[0, i, :, :] #신뢰도얻는과정

        _, prob, _, point = cv2.minMaxLoc(probMap)

        x = int(imageWidth * point[0] / W)
        y = int(imageHeight * point[1] / H)

        if prob > 0.1:
            points.append((x, y))
        else:
            points.append(None)

    partA = BODY_PARTS["RKnee"]   
    partB = BODY_PARTS["Chest"]   
    partC = BODY_PARTS["LKnee"] 

    if points[partA] and points[partB]:
        cv2.line(frame, points[partA], points[partB], (0, 255, 0), 2)

    if points[partC] != None:
        if points[partB] and points[partC]:
            cv2.line(frame, points[partB], points[partC], (0, 255, 0), 2)

    if points[partC] != None:
        if isinstance(points[partA], tuple) and isinstance(points[partB], tuple):
            x1, y1 = points[partA] 
            x2, y2 = points[partB] 
            x3, y3 = points[partC]

            cv2.circle(frame, (x1, y1), 4, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
            cv2.circle(frame, (x2, y2), 4, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
            cv2.circle(frame, (x3, y3), 4, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)

            angle = np.arctan2(y3 - y2, x3 - x2) - np.arctan2(y1 - y2, x1 - x2)
            angle = np.degrees(angle)
            if angle < 0:
                angle += 360
            if angle > 180:
                angle -= 360
                angle = angle * -1
            #print(angle)
            if angle > 60: #빨강
                cv2.line(frame, points[partA], points[partB], (0, 0, 255), 2)
                cv2.line(frame, points[partB], points[partC], (0, 0, 255), 2)
            if angle < 30: 
                cv2.line(frame, points[partA], points[partB], (0, 0, 255), 2)
                cv2.line(frame, points[partB], points[partC], (0, 0, 255), 2)

    return frame


def capture_frame():
    frame = generate_virtual_frame()

    ret, buffer = cv2.imencode('.jpg', frame)
    frame_bytes = buffer.tobytes()

    return frame_bytes


def generate_frames():

    last_frame_time = time.time()

    while True:
        frame = capture_frame()

        current_time = time.time()

        if current_time - last_frame_time < 0.01:  # 0.01초 설정
            continue

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        
        last_frame_time = current_time


@app.route('/')
def index():
    return render_template('waist.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)


