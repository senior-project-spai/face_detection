# import the necessary packages
from imutils.video import VideoStream
from imutils import face_utils
import datetime
import argparse
import os
import imutils
import time
import dlib
import cv2
from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests
import json

usePiCamera = False

BRANCH_ID = 0
CAMERA_ID = 0

vs = None
best_frame = None
best_frame_reduced = None
max_confidence = -1
best_frame_buttom = -1
best_frame_right = -1
best_frame_top = -1
best_frame_left = -1
best_frame_size = -1
pictureName = ""

app = FastAPI()
# app.add_middleware(CORSMiddleware, allow_origins=['*'])
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class Idetection_response(BaseModel):
    face_image_id: str
    photo_uri: str
    confidence: float
    buttom: float
    right: float
    top: float
    left: float
    size: float


@app.on_event("startup")
def startup_event():
    global vs
    # initialize the video stream and allow the camera sensor to warm up
    print("[INFO] camera sensor warming up...")
    vs = VideoStream(usePiCamera=usePiCamera, resolution=(960, 720)).start()
    time.sleep(2.0)


@app.on_event("shutdown")
def shutdown_event():
    vs.stop()


def get_width_frame(frame):
    (_, w) = frame.shape[:2]
    return w


def get_height_frame(frame):
    (h, _) = frame.shape[:2]
    return h


def calculate_det_size(det):
    return (det.bottom()-det.top()) * (det.right()-det.left())


def get_biggest_face(dets, scores, idx):
    biggest_det_det = -1
    biggest_det_score = -1
    biggest_det_idx = -1
    biggest_det_size = -1
    for i, d in enumerate(dets):
        size = calculate_det_size(d)
        if size > biggest_det_size:
            biggest_det_size = size
            biggest_det_idx = idx[i]
            biggest_det_score = scores[i]
            biggest_det_det = d
    return biggest_det_det, biggest_det_score, biggest_det_idx, biggest_det_size


def upload_to_face_input_api(frame):
    # Uploading to S3
    print("Uploading Frame to Image Input API")
    currentTime = int(round(time.time() * 1000)) / 1000
    global pictureName
    global BRANCH_ID
    global CAMERA_ID
    pictureName = "FaceDetector_" + \
        str(BRANCH_ID)+"_"+str(CAMERA_ID)+"_"+str(currentTime)+".jpg"
    data = {
        'time': currentTime,
        'branch_id': BRANCH_ID,
        'camera_id': CAMERA_ID,
        'image_name': pictureName,
        "position_bottom": best_frame_buttom,
        "position_right": best_frame_right,
        "position_top": best_frame_top,
        "position_left": best_frame_left,
    }
    file = {'image': (pictureName, frame.tostring(),
                      'image/jpeg', {'Expires': '0'})}
    response = requests.post(
        "https://face-image-input-api-spai.apps.spai.ml/_api/face", files=file, data=data)

    return json.loads(response.text)


def detection(detector, frame):
    # The score is bigger for more confident detections.
    dets, scores, idx = detector.run(frame, 0)
    det, score, idx, size = get_biggest_face(dets, scores, idx)

    # det == -1 means not found face in picture
    if det != -1:
        print("Left: {} Top: {} Right: {} Bottom: {} IDX:{} Score:{} Size:{}".format(
            det.left(), det.top(), det.right(), det.bottom(), idx, score, size))
        response = {
            'left': det.left(),
            'top': det.top(),
            'right': det.right(),
            'bottom': det.bottom(),
            'score': score,
            'size': size
        }
        return response
    else:
        print("Face Not Found")
        return {}


def detections(detector, frame):
    global best_frame
    global best_frame_reduced
    global max_confidence
    global best_frame_buttom
    global best_frame_right
    global best_frame_top
    global best_frame_left
    global best_frame_size

    det = detection(detector, frame)
    if det != {}:
        if det['score'] > max_confidence:
            print("[INFO] BEST FRAME")
            best_frame = frame
            max_confidence = det['score']
            best_frame_buttom = det['bottom']
            best_frame_right = det['right']
            best_frame_top = det['top']
            best_frame_left = det['left']
            best_frame_size = det['size']
    else:
        print("Face Not Found")


@app.get("/detection", response_model=Idetection_response)
async def trigger_detection():
    global best_frame
    global best_frame_reduced
    global max_confidence
    global best_frame_buttom
    global best_frame_right
    global best_frame_top
    global best_frame_left
    global best_frame_size

    best_frame = None
    best_frame_reduced = None
    max_confidence = -1
    best_frame_buttom = -1
    best_frame_right = -1
    best_frame_top = -1
    best_frame_left = -1
    best_frame_size = -1

    # initialize dlib's face detector (HOG-based) and
    # using default detector from dlib
    print("[INFO] major face detector...")
    detector = dlib.get_frontal_face_detector()

    # loop frame by frame from video stream
    i = 0
    while True:
        # grab the frame from the threaded video stream
        if i >= 50 and best_frame is not None:
            break
        print("[INFO] "+str(i)+": ", end="")
        frame = vs.read()
        frame = imutils.resize(frame, width=400)
        detections(detector, frame)
        i += 1

    # Save Best frame
    cv2.imwrite("Best.jpg", best_frame)
    _, best_frame_encoded = cv2.imencode('.jpg', best_frame)
    det = detection(detector, best_frame)
    max_confidence = det['score']
    best_frame_buttom = det['bottom']
    best_frame_right = det['right']
    best_frame_top = det['top']
    best_frame_left = det['left']
    best_frame_size = det['size']
    print("[INFO] Best Frame Left: {} Top: {} Right: {} Bottom: {} Score:{} Size:{}".format(
        best_frame_left, best_frame_top, best_frame_right, best_frame_buttom, max_confidence, best_frame_size))

    response = upload_to_face_input_api(best_frame_encoded)

    photo_uri = None
    while True:
        try:
            print('[INFO] Retrieve photo from S3 Server')
            photo_uri = requests.get(
                url='https://get-photo-from-s3-spai.apps.spai.ml/_api/photo/'+pictureName).json()['photo_data_uri']
        except:
            continue
        break
    return {
        "confidence": max_confidence,
        "buttom": best_frame_buttom,
        "right": best_frame_right,
        "top": best_frame_top,
        "left": best_frame_left,
        "size": best_frame_size,
        'face_image_id': response['face_image_id'],
        'photo_uri': photo_uri
    }
