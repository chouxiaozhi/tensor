import cv2
import requests
import numpy as np
import json
import os
import multiprocessing
# 定义目标类别
CLASSES = ['car', 'trunk']


def preprocess_image(image, input_size=(320, 320)):
    # Resize image to model input size
    img = cv2.resize(image, input_size)
    img = img[:, :, ::-1].transpose(2, 0, 1)  # Convert BGR to RGB, and reorder to CHW
    img = np.expand_dims(img, axis=0).astype(np.float32)
    img /= 255.0  # Normalize to [0,1]
    return img


def send_frame_to_tf_serving(image, server_url):
    # Preprocess image for TensorFlow Serving
    img = preprocess_image(image)

    # Create the data payload for TensorFlow Serving
    data = json.dumps({
        "signature_name": "serving_default",  # Depends on your model signature
        "instances": img.tolist()
    })

    headers = {"content-type": "application/json"}

    # Send POST request to TensorFlow Serving
    json_response = requests.post(server_url, data=data, headers=headers)

    # print("Response status code:", json_response.status_code)
    # print("Response text:", json_response.text)
    # Check response status
    if json_response.status_code != 200:
        print(f"Error: Received response with status code {json_response.status_code}")
        print("json_response.text",json_response.text)
        return None

    # Get predictions from the response
    response_data = json.loads(json_response.text)
    # print("Response from server:", response_data)
    with open('response_data.json', 'w') as f:
        json.dump(response_data['predictions'], f, indent=4)
    if 'predictions' not in response_data:
        print("Error: 'predictions' not found in response")
        return None

    # Get predictions from the response
    # predictions = json.loads(json_response.text)['predictions']

    predictions = response_data['predictions']
    return predictions


def draw_predictions(image, boxes, classes, scores, threshold=0.5):
    for i, box in enumerate(boxes):
        if scores[i] < threshold:
            continue
        top, left, right, bottom = box
        cl = int(classes[i])
        label = CLASSES[cl]
        cv2.rectangle(image, (top, left), (right, bottom), (0, 255, 0), 2)
        cv2.putText(image, f'{label} {scores[i]:.2f}', (top, left), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
    return image


def video_processing_process(video_path, server_url, process_id):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        # print(f"Error: Could not open video {video_path}.")
        return

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Send frame to TensorFlow Serving and get predictions
            predictions = send_frame_to_tf_serving(frame, server_url)

            if predictions is None:
                continue  # Skip this frame if there was an error
            # Print predictions for debugging
            # print("Predictions received:", predictions)

            try:
                # print('predictions',predictions)
                boxes = predictions[0]['onnx::Sigmoid_369']
                classes = predictions[0]['onnx::Sigmoid_468']
                scores = predictions[0]['onnx::Sigmoid_420']
            except KeyError as e:
                print(f"Error extracting prediction data: {e}")
                continue  # Skip this frame if there was an error

            # Draw predictions on the frame
            frame = draw_predictions(frame, boxes, classes, scores)

            # Display the frame
            cv2.imshow(f'YOLOv5 Detection - {os.path.basename(video_path)}', frame)

            # Quit if 'q' is pressed
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
    # for video_path in video_paths:
    #     process = multiprocessing.Process(target=video_processing_process, args=(video_path, server_url))
    #     video_processes.append(process)
    #     process.start()
    #
    # for process in video_processes:
    #     process.join()

    for idx, video_path in enumerate(video_paths):
        process_id = f"Process{idx + 1}"
        process = multiprocessing.Process(target=video_processing_process,
                                          args=(video_path, server_url, process_id))
        video_processes.append(process)
        process.start()

    for process in video_processes:
        process.join()
