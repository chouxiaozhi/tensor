import cv2
import requests
import numpy as np
import json
import os
import multiprocessing

# 定义目标类别
CLASSES = ['car', 'trunk']

def preprocess_image(image, input_size=(320, 320)):
    img = cv2.resize(image, input_size)
    img = img[:, :, ::-1].transpose(2, 0, 1)  # Convert BGR to RGB
    img = np.expand_dims(img, axis=0).astype(np.float32) / 255.0  # Normalize
    return img

def send_frame_to_tf_serving(image, server_url):
    img = preprocess_image(image)

    data = json.dumps({
        "signature_name": "serving_default",
        "instances": img.tolist()
    })

    headers = {"content-type": "application/json"}
    json_response = requests.post(server_url, data=data, headers=headers)

    if json_response.status_code != 200:
        print("Error: Received non-200 response:", json_response.status_code)
        return None

    response_text = json_response.text
    try:
        predictions = json.loads(response_text)['predictions']
    except (json.JSONDecodeError, KeyError) as e:
        print("Error extracting prediction data:", e)
        return None

    return predictions

def draw_predictions(image, boxes, classes, scores, threshold=0.5):
    for i, box in enumerate(boxes):
        if scores[i] < threshold:
            continue
        top, left, right, bottom = box
        cl = int(classes[i])
        label = CLASSES[cl]
        cv2.rectangle(image, (int(left), int(top)), (int(right), int(bottom)), (0, 255, 0), 2)
        cv2.putText(image, f'{label} {scores[i]:.2f}', (int(left), int(top)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
    return image

def video_processing_process(video_path, server_url):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}.")
        return

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            predictions = send_frame_to_tf_serving(frame, server_url)
            if predictions is None:
                continue

            print(predictions)

            try:
                boxes = predictions[0]['detection_boxes']
                classes = predictions[0]['detection_classes']
                scores = predictions[0]['detection_scores']
            except KeyError as e:
                print(f"Error extracting prediction data: {e}")
                continue  # Skip this frame if there was an error

            frame = draw_predictions(frame, boxes, classes, scores)

            cv2.imshow(f'YOLOv5 Detection - {os.path.basename(video_path)}', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    server_url = 'http://192.168.47.3:8501/v1/models/yolo5_model:predict'
    video_paths = ['highway4.mp4', 'highway.mp4']

    video_processes = []
    for video_path in video_paths:
        process = multiprocessing.Process(target=video_processing_process, args=(video_path, server_url))
        video_processes.append(process)
        process.start()

    for process in video_processes:
        process.join()
