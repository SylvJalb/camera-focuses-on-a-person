# Focus a person
Focus a person in a live video stream or a video file.

## Description
The YOLOv10s model is used to detect people in a video. This program return the position of the most centered person in the frame. The position is returned as a percentage of the frame width and height from the center. The program can be used to focus a camera on a person in a video stream or a video file.

## Installation
Run the following command into a virtual environment.
```
pip install -r requirements.txt
```

## Usage
```
python run.py [-h] [--video-path VIDEO_PATH] [--model MODEL] [--show-result]
```