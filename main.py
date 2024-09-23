import os
import cv2
import numpy as np
import onnxruntime
import time
import serial
import threading
import multiprocessing

'''
输入名字叫 images
input_name ['images']
输出名字叫  这些
output_name ['output', 'onnx::Sigmoid_369', 'onnx::Sigmoid_420', 'onnx::Sigmoid_468']
input_name ['images']
output_name ['output', 'onnx::Sigmoid_369', 'onnx::Sigmoid_420', 'onnx::Sigmoid_468']
'''
# 定义目标类别

CLASSES = ['car', 'trunk']

#模型文件
class YOLOV5:
    def __init__(self, onnxpath):
        self.onnx_session = onnxruntime.InferenceSession(onnxpath)
        self.input_name = self.get_input_name()
        self.output_name = self.get_output_name()

    def get_input_name(self):
        input_name = []
        for node in self.onnx_session.get_inputs():
            input_name.append(node.name)
        # print('input_name',input_name)
        return input_name

    def get_output_name(self):
        output_name = []
        for node in self.onnx_session.get_outputs():
            output_name.append(node.name)
        # print('output_name', output_name)
        return output_name

    def get_input_feed(self, img_tensor):
        input_feed = {}
        for name in self.input_name:
            input_feed[name] = img_tensor
        # print('input_feed',input_feed)
        return input_feed

    def inference(self, img):        #
        or_img = cv2.resize(img, (320, 320))
        img = or_img[:, :, ::-1].transpose(2, 0, 1) # 把channel
        img = img.astype(dtype=np.float32)
        img /= 255.0
        img = np.expand_dims(img, axis=0)
        # print(img.shape)
        input_feed = self.get_input_feed(img)
        pred = self.onnx_session.run(None, input_feed)[0]  #进行预测  这只得就是四个输出 四个变量
        return pred, or_img


def nms(dets, thresh):
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    areas = (y2 - y1 + 1) * (x2 - x1 + 1)
    scores = dets[:, 4]
    keep = []
    index = scores.argsort()[::-1]

    while index.size > 0:
        i = index[0]
        keep.append(i)
        x11 = np.maximum(x1[i], x1[index[1:]])
        y11 = np.maximum(y1[i], y1[index[1:]])
        x22 = np.minimum(x2[i], x2[index[1:]])
        y22 = np.minimum(y2[i], y2[index[1:]])

        w = np.maximum(0, x22 - x11 + 1)
        h = np.maximum(0, y22 - y11 + 1)

        overlaps = w * h
        ious = overlaps / (areas[i] + areas[index[1:]] - overlaps)
        idx = np.where(ious <= thresh)[0]
        index = index[idx + 1]
    return keep


def xywh2xyxy(x):
    y = np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2
    y[:, 1] = x[:, 1] - x[:, 3] / 2
    y[:, 2] = x[:, 0] + x[:, 2] / 2
    y[:, 3] = x[:, 1] + x[:, 3] / 2
    return y


def filter_box(org_box, conf_thres, iou_thres):
    org_box = np.squeeze(org_box)
    conf = org_box[..., 4] > conf_thres
    box = org_box[conf == True]

    if len(box) == 0:
        return np.array([])

    cls_cinf = box[..., 5:]
    cls = []
    for i in range(len(cls_cinf)):
        cls.append(int(np.argmax(cls_cinf[i])))
    all_cls = list(set(cls))

    output = []
    for i in range(len(all_cls)):
        curr_cls = all_cls[i]
        curr_cls_box = []
        for j in range(len(cls)):
            if cls[j] == curr_cls:
                box[j][5] = curr_cls
                curr_cls_box.append(box[j][:6])
        curr_cls_box = np.array(curr_cls_box)
        curr_cls_box = xywh2xyxy(curr_cls_box)
        curr_out_box = nms(curr_cls_box, iou_thres)
        for k in curr_out_box:
            output.append(curr_cls_box[k])
    output = np.array(output)
    return output

def draw(image, box_data):
    if box_data.size == 0:
        return

    boxes = box_data[..., :4].astype(np.int32)
    scores = box_data[..., 4]
    classes = box_data[..., 5].astype(np.int32)

    for box, score, cl in zip(boxes, scores, classes):
        top, left, right, bottom = box
        cv2.rectangle(image, (top, left), (right, bottom), (0, 255, 0), 2)
        # cv2.putText(image, '{0} {1:.2f}'.format(CLASSES[cl], score),
        # (top, left),
        # cv2.FONT_HERSHEY_SIMPLEX,
        # 0.6, (0, 0, 255), 1)


# 这里改成我直接给你输出参数
def video_processing_process(video_path, onnx_path, detection_event, process_id):
    # 在子进程中初始化模型
    model = YOLOV5(onnxpath=onnx_path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = 0
    start_time = time.time()

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            output, or_img = model.inference(frame)

            # print('output',output)
            outbox = filter_box(output, 0.45, 0.3)  # 放入原始输出
            draw(or_img, outbox)

            detected_truck = any(
                CLASSES[int(cls)] in ['car', 'trunk'] for cls in outbox[:, 5]) if outbox.size > 0 else False
            if detected_truck:
                detection_event.set()
            else:
                detection_event.clear()

            frame_count += 1
            elapsed_time = time.time() - start_time
            current_fps = frame_count / elapsed_time

            cv2.putText(or_img, f'FPS: {current_fps:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            or_img = cv2.resize(or_img, (640, 640))
            cv2.imshow(f'YOLOv5 Detection - {os.path.basename(video_path)}', or_img)

            print(f"[{process_id}] Detected truck: {detected_truck}")

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()
        detection_event.clear()


if __name__ == "__main__":
    onnx_path = 'yolov5_highway_n_320.onnx'  # 这个是模型
    video_paths = ['highway4.mp4', 'highway.mp4']  # 视频的路径
    # model = YOLOV5(onnxpath=onnx_path)
    detection_event = multiprocessing.Event()
    video_processes = []
    for idx, video_path in enumerate(video_paths):
        process_id = f"Process{idx + 1}"
        process = multiprocessing.Process(target=video_processing_process,
                                          args=(video_path, onnx_path, detection_event, process_id))
        video_processes.append(process)
        process.start()

    for process in video_processes:
        process.join()

    detection_event.clear()

