# How to run

## Run with Raspberry Pi
### Install prerequisite
```
sudo apt install python-opencv libatlas-base-dev libjasper-dev libqtgui4 python3-pyqt5 libqt4-test
pip3 install imutils opencv-python opencv-contrib-python==4.1.0.25 uvicorn
```
### Connect to VPN KU
```
sudo openvpn --config path/to/.ovpnfile --daemon
```
### if you use pi camera
```
set usePiCamera to True in code
```
### if you don't use pi camera
```
set usePiCamera to False in code
```
## Run in development
```
uvicorn face_detector:app --reload --host 0.0.0.0 --port 8002
```
## Run in production
```
uvicorn face_detector:app --host 0.0.0.0
```
## Credit 
https://www.pyimagesearch.com/2017/04/17/real-time-facial-landmark-detection-opencv-python-dlib/
