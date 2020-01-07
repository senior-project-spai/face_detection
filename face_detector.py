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
from pydantic import BaseModel
import requests
import json
usePiCamera = False

app = FastAPI()


class Itrigger_detection(BaseModel):
    customerId: str = None
    transactionId: str = None


class Idetection_response(BaseModel):
    customerId: str
    transactionId: str
    confidence: float
    buttom: float
    right: float
    top: float
    left: float
    size: float


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


def uploadS3(frame):
    # Uploading to S3
    print("Uploading Frame to S3")

    data = {
        'time': int(time.time()),
        'branch_id': 123456,
        'camera_id': 123456,
        'picture': frame.tostring(),
        'pictureName': 'Best-test-123456.jpg',
    }

    response = requests.post(
        "https://image-to-s3-spai.apps.spai.ml/_api/image", data=data)

    print(json.loads(response.text))


@app.post("/detection", response_model=Idetection_response)
async def trigger_detection(body: Itrigger_detection):
    # initialize dlib's face detector (HOG-based) and
    # using default detector from dlib
    print("[INFO] major face detector...")
    detector = dlib.get_frontal_face_detector()

    # initialize the video stream and allow the camera sensor to warm up
    print("[INFO] camera sensor warming up...")

    vs = VideoStream(usePiCamera=usePiCamera, resolution=(960, 720)).start()
    time.sleep(2.0)

    best_frame = None
    best_frame_reduced = None
    max_confidence = -1
    best_frame_buttom = -1
    best_frame_right = -1
    best_frame_top = -1
    best_frame_left = -1
    best_frame_size = -1

    # loop frame by frame from video stream
    for _ in range(0, 50):
        # grab the frame from the threaded video stream,
        # resize it to maximum width of 400 pixels
        frame = vs.read()
        raw_frame = frame
        frame = imutils.resize(frame, width=400)
        raw_frame_reduced = frame

        h, w = get_height_frame(frame), get_width_frame(frame)

        # The score is bigger for more confident detections.
        dets, scores, idx = detector.run(frame, 0)
        det, score, idx, size = get_biggest_face(dets, scores, idx)

        # det == -1 means not found face in picture
        if det != -1:
            print("Left: {} Top: {} Right: {} Bottom: {} IDX:{} Score:{} Size:{}".format(
                det.left(), det.top(), det.right(), det.bottom(), idx, score, size))

            if score > max_confidence:
                print("BEST!!!!!!!!!!!!!")
                best_frame = raw_frame
                best_frame_reduced = raw_frame_reduced
                max_confidence = score
                best_frame_buttom = det.bottom()
                best_frame_right = det.right()
                best_frame_top = det.top()
                best_frame_left = det.left()
                best_frame_size = size
        else:
            print("Face Not Found")

    # Save Best frame
    cv2.imwrite("Best.jpg", best_frame)
    cv2.imwrite("Best_reduced.jpg", best_frame_reduced)
    _, best_frame_encoded = cv2.imencode('.jpg', best_frame)
    print("Best Frame Left: {} Top: {} Right: {} Bottom: {} Score:{} Size:{}".format(
        best_frame_left, best_frame_top, best_frame_right, best_frame_buttom, max_confidence, best_frame_size))

    # Cleanup
    vs.stop()

    uploadS3(best_frame_encoded)

    return {
        "customerId": body.customerId,
        "transactionId": body.transactionId,
        "confidence": max_confidence,
        "buttom": best_frame_buttom,
        "right": best_frame_right,
        "top": best_frame_top,
        "left": best_frame_left,
        "size": best_frame_size,
    }
