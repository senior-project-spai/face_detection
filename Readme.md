# How to run

## Run with Raspberry Pi
### Install prerequisite
```
sudo apt install python-opencv libatlas-base-dev libjasper-dev libqtgui4 python3-pyqt5 libqt4-test
pip3 install imutils opencv-python opencv-contrib-python==4.1.0.25
```
### if you use pi camera
```
python3 face_detector.py --picamera 1
```
## Run in development
```
uvicorn face_detector:app --reload
```
## Run in production
```
uvicorn face_detector:app
```
## Credit 
https://www.pyimagesearch.com/2017/04/17/real-time-facial-landmark-detection-opencv-python-dlib/
