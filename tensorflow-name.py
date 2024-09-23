import tensorflow as tf

# 加载模型
model = tf.saved_model.load('yolo5_model/1')

# 查看签名
signatures = model.signatures

# 输出默认签名的输入和输出
for name, sig in signatures.items():
    print(f"Signature: {name}")
    print(f"Input(s): {sig.structured_input_signature}")
    print(f"Output(s): {sig.structured_outputs}")\

'''
Signature: serving_default
Input(s): ((), {'images': TensorSpec(shape=(1, 3, 320, 320), dtype=tf.float32, name='images')})
Output(s): {
'onnx::Sigmoid_420': TensorSpec(shape=(1, 3, 20, 20, 7), dtype=tf.float32, name='onnx__sigmoid_420'), 
'onnx::Sigmoid_468': TensorSpec(shape=(1, 3, 10, 10, 7), dtype=tf.float32, name='onnx__sigmoid_468'), 
'output': TensorSpec(shape=(1, 6300, 7), dtype=tf.float32, name='output'), 
'onnx::Sigmoid_369': TensorSpec(shape=(1, 3, 40, 40, 7), dtype=tf.float32, name='onnx__sigmoid_369')}
'''