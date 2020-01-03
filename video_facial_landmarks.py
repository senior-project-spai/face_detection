# import the necessary packages
from imutils.video import VideoStream
from imutils import face_utils
import datetime
import argparse
import imutils
import time
import dlib
import cv2


def calculate_size(det):
    return (det.bottom()-det.top()) * (det.right()-det.left())


def get_biggest_face(dets, scores, idx):
    biggest_det_det = -1
    biggest_det_score = -1
    biggest_det_idx = -1
    biggest_det_size = -1
    for i, d in enumerate(dets):
        size = calculate_size(d)
        if size > biggest_det_size:
            biggest_det_size = size
            biggest_det_idx = idx[i]
            biggest_det_score = scores[i]
            biggest_det_det = d
    # -1 means not found face in picture
    if biggest_det_idx == -1 or biggest_det_size == -1 or biggest_det_det == -1 or biggest_det_score == -1:
        print("Face Not Found")
    return biggest_det_det, biggest_det_score, biggest_det_idx, biggest_det_size


if __name__ == "__main__":
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-r", "--picamera", type=int, default=-1,
                    help="whether or not the Raspberry Pi camera should be used")
    args = vars(ap.parse_args())

    # initialize dlib's face detector (HOG-based) and then create
    # the facial landmark predictor
    print("[INFO] loading facial landmark predictor...")
    detector = dlib.get_frontal_face_detector()
    # predictor = dlib.shape_predictor(args["shape_predictor"])

    # initialize the video stream and allow the cammera sensor to warmup
    print("[INFO] camera sensor warming up...")
    vs = VideoStream(usePiCamera=args["picamera"] > 0).start()
    time.sleep(2.0)

    # loop over the frames from the video stream
    while True:
        # grab the frame from the threaded video stream, resize it to
        # have a maximum width of 400 pixels, and convert it to
        # grayscale
        frame = vs.read()
        frame = imutils.resize(frame, width=400)
        # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        dets, scores, idx = detector.run(frame, 0)
        det, score, idx, size = get_biggest_face(dets, scores, idx)
        if det != -1:
            font = cv2.FONT_HERSHEY_DUPLEX
            text_showed = "{} {:0.2f} {:0.2f}".format(idx, score, size)
            print("Left: {} Top: {} Right: {} Bottom: {} IDX:{} Score:{} Size:{}".format(
                det.left(), det.top(), det.right(), det.bottom(), idx, score, size))
            cv2.rectangle(frame, (det.left(), det.top()),
                        (det.right(), det.bottom()), (0, 0, 255), 2)
            cv2.putText(frame, text_showed, (det.left() + 6, det.bottom() - 6),
                        font, 0.5, (255, 255, 255), 1)

        # show the frame
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF
        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            print("Exit")
            break

    # do a bit of cleanup
    cv2.destroyAllWindows()
    vs.stop()
