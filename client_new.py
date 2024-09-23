import cv2
import requests
import numpy as np
import json
import os
import multiprocessing

# 定义目标类别
CLASSES = ['car', 'trunk']

#图像预处理
def preprocess_image(image, input_size=(320, 320)):
    # Resize image to model input size
    img = cv2.resize(image, input_size)
    img = img[:, :, ::-1].transpose(2, 0, 1)  # Convert BGR to RGB, and reorder to CHW
    img = img.astype(dtype=np.float32)
    img /= 255.0  # Normalize to [0,1]
    img = np.expand_dims(img, axis=0)
    return img
# 这个就是获得了
def send_frame_to_tf_serving(image, server_url):
    # 进行预处理
    img = preprocess_image(image)
    data = json.dumps({
        "signature_name": "serving_default",  # Depends on your model signature
        "instances": img.tolist() # 将图片数据传入
    })
    headers = {"content-type": "application/json"}

    json_response = requests.post(server_url, data=data, headers=headers)

    if json_response.status_code != 200:
        print(f"Error: Received response with status code {json_response.status_code}")
        print("json_response.text", json_response.text)
        return None

    # Get predictions from the response
    response_data = json.loads(json_response.text)

    if 'predictions' not in response_data:
        print("Error: 'predictions' not found in response")
        return None

    predictions = response_data['predictions']
    return predictions[0]

def video_processing_process(video_path, server_url):
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

    except KeyboardInterrupt:
        pass
    finally:
        cap.release()


if __name__ == "__main__":
    server_url = 'http://192.168.47.3:8501/v1/models/yolo5_model:predict'
    video_paths = ['highway4.mp4', 'highway.mp4']
    for idx, video_path in enumerate(video_paths):
        video_processing_process(video_path, server_url)


