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

usePiCamera = False

app = FastAPI()


class Itrigger_detection(BaseModel):
    name: str = None
    description: str = None
    price: float = None
    tax: float = None


class Idetection_response(BaseModel):
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


@app.post("/detection", response_model=Idetection_response)
async def trigger_detection(item: Itrigger_detection):
    # construct the argument parse and parse the arguments

    # ap = argparse.ArgumentParser()
    # ap.add_argument("-r", "--picamera", type=int, default=-1,
    #                 help="whether or not the Raspberry Pi camera should be used")
    # args = vars(ap.parse_args())

    # initialize dlib's face detector (HOG-based) and
    # using default detector from dlib
    print("[INFO] major face detector...")
    detector = dlib.get_frontal_face_detector()

    # initialize the video stream and allow the camera sensor to warm up
    print("[INFO] camera sensor warming up...")

    vs = VideoStream(usePiCamera=usePiCamera).start()
    time.sleep(1.0)
    no_face_count = 0

    best_frame = None
    best_frame_reduced = None
    max_confidence = -1
    best_frame_buttom = -1
    best_frame_right = -1
    best_frame_top = -1
    best_frame_left = -1
    best_frame_size = -1

    # loop frame by frame from video stream
    for _ in range(0, 200):
        # grab the frame from the threaded video stream,
        # resize it to maximum width of 400 pixels
        frame = vs.read()
        raw_frame = frame
        frame = imutils.resize(frame, width=400)
        raw_frame_reduced = frame
        # Convert to grayscale (for performance purpose i think)
        # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        h, w = get_height_frame(frame), get_width_frame(frame)

        # The score is bigger for more confident detections.
        dets, scores, idx = detector.run(frame, 0)
        det, score, idx, size = get_biggest_face(dets, scores, idx)

        # det == -1 means not found face in picture
        if det != -1:
            # no_face_count = 0
            # font = cv2.FONT_HERSHEY_DUPLEX
            # text_showed = "{} {:0.2f} {:0.2f}".format(idx, score, size)
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

            # cv2.rectangle(frame, (det.left(), det.top()),
            #               (det.right(), det.bottom()), (0, 0, 255), 2)
            # cv2.putText(frame, text_showed, (det.left() + 6, det.bottom() - 6),
            #             font, 0.5, (255, 255, 255), 1)
        # else:
            # no_face_count = no_face_count+1
            # print("Face Not Found Count={}".format(no_face_count))

        # this if scope is to call api that next frame should be new customer
        # if no_face_count == 15:
            # TODO: Create function to tell that its new client coming to store
            # print('New Client')

        # show the frame
        cv2.imshow("Frame", frame)
        # key = cv2.waitKey(1) & 0xFF

        # if the `q` key was pressed, break from the loop
        # if key == ord("q"):
        #     print("Exit")
        #     break

    # Show Best frame
    cv2.imwrite("Best.jpg", best_frame)
    cv2.imwrite("Best_reduced.jpg", best_frame_reduced)
    print("Best Frame Left: {} Top: {} Right: {} Bottom: {} Score:{} Size:{}".format(
        best_frame_left, best_frame_top, best_frame_right, best_frame_buttom, max_confidence, best_frame_size))
    # font = cv2.FONT_HERSHEY_DUPLEX
    # text_showed = "{:0.2f} {:0.2f}".format(max_confidence, best_frame_size)
    # cv2.rectangle(best_frame, (best_frame_left, best_frame_top),
    #               (best_frame_right, best_frame_buttom), (0, 0, 255), 2)
    # cv2.putText(best_frame, text_showed, (best_frame_left + 6, best_frame_buttom - 6),
    #             font, 0.5, (255, 255, 255), 1)

    # cv2.imshow("Best Frame", best_frame)

    # Cleanup
    # cv2.destroyAllWindows()
    vs.stop()
    time.sleep(2)
    return {
        "confidence": max_confidence,
        "buttom": best_frame_buttom,
        "right": best_frame_right,
        "top": best_frame_top,
        "left": best_frame_left,
        "size": best_frame_size,
    }
