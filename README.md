# Focus a person
Calculate the position of the most centered person in a video frame. The program can be used to focus a person in a live video stream or a video file.  

![GIF](./drone_test_result.gif)     
(Video source: https://www.kaggle.com/datasets/kmader/drone-videos?select=Berghouse+Leopard+Jog.mp4)

## Description
The YOLOv10s model is used to detect people in a video. This program return the distance from the center of the most centered person in the frame. The program can be used to focus a person and turn a camera or a drone to follow the person.

## Installation
Run the following command into a virtual environment.
```
pip install -r requirements.txt
```

## Usage
```
python run.py [-h] [--video-path VIDEO_PATH] [--model MODEL] [--show-result]
```

## Performance
YoloV10s model have 7.2 million parameters, and need 21.6 GFLOPs. The latency is only 2,49 ms.

## Improvement
- [ ] Add an histeresis to avoid the little changement of focused person.
- [ ] Do batch processing to improve the performance.