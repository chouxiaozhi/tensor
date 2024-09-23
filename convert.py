import onnx
from onnx_tf.backend import prepare

# 加载 ONNX 模型
onnx_model = onnx.load("yolov5_highway_n_320.onnx")  # 替换成你的 onnx 模型文件路径

# 将 ONNX 模型转换为 TensorFlow 格式
tf_rep = prepare(onnx_model)

# 保存为 TensorFlow 的模型
tf_rep.export_graph("save_model")  # 替换为你想保存的 TensorFlow 模型的路径
