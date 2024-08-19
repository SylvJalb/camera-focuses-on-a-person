import argparse
import cv2
from ultralytics import YOLOv10

def process_video(video_path, model, show_result):
    cap = cv2.VideoCapture(video_path)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        output = model.predict(frame)
        if show_result:
            display_output(frame, output)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    

def display_output(frame, output):
    if len(output) > 0:
        # Afficher les résultats sur la vidéo
        labels = output[0].names
        for boxes in output[0].boxes:
            label = labels[int(boxes.cls)]
            for box in boxes.xyxy:
                x1, y1, x2, y2 = box
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (36,255,12), 2)
                cv2.putText(frame, label, (int(x1), int(y1-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
    # Afficher la frame
    cv2.imshow("Result", frame)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Person focus")

    # Video path
    parser.add_argument("--video-path",
                        type=str,
                        help="Path to the video file. If not provided, webcam will be used.",
                        default=0)
    
    parser.add_argument("--model",
                        type=str,
                        help="Model name.",
                        default='jameslahm/yolov10s')
    
    parser.add_argument("--show-result",
                        action="store_true",
                        help="Show the result of the prediction.")

    args = parser.parse_args()

    # Load model
    model = YOLOv10.from_pretrained(args.model)

    # Process webcam feed
    process_video(args.video_path, model, args.show_result)