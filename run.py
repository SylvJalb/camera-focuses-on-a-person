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
        persons = filter_persons(output)
        person = get_the_most_central_person(persons, frame.shape[1], frame.shape[0])
        if show_result:
            display_result(frame, person)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

def filter_persons(output):
    # Filtrer les résultats pour obtenir uniquement les personnes
    for boxes in output[0].boxes:
        if boxes.cls == 0:
            return boxes
    return None

def get_the_most_central_person(persons, image_width, image_height):
    # Obtenir la personne la plus centrale
    best_person = None
    best_distance = image_width + image_height
    for person in persons.xyxy:
        x1, y1, x2, y2 = person
        x_center, y_center = (x1 + x2) / 2, (y1 + y2) / 2
        x_distance = abs(image_width / 2 - x_center)
        y_distance = abs(image_height / 2 - y_center)
        distance = x_distance + y_distance
        if distance < best_distance:
            # On a trouvé une personne plus centrale
            best_person = person
            best_distance = distance
    return best_person
    

def display_result(frame, person):
    if person is not None:
        # Afficher la personne sur la vidéo
        x1, y1, x2, y2 = person
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (36,255,12), 2)
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

    # Process webcam
    process_video(args.video_path, model, args.show_result)