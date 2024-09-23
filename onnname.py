import onnx

# 加载 ONNX 模型
model = onnx.load('yolov5_highway_n_320.onnx')

# 获取模型的输入信息
input_info = []
for input in model.graph.input:
    input_info.append({
        'name': input.name,
        'shape': [dim.dim_value for dim in input.type.tensor_type.shape.dim],
        'dtype': input.type.tensor_type.elem_type
    })

# 获取模型的输出信息
output_info = []
for output in model.graph.output:
    output_info.append({
        'name': output.name,
        'shape': [dim.dim_value for dim in output.type.tensor_type.shape.dim],
        'dtype': output.type.tensor_type.elem_type
    })

# 打印输入和输出的信息
print("Input Info:")
for info in input_info:
    print(info)

print("\nOutput Info:")
for info in output_info:
    print(info)
'''
Input Info:
{'name': 'images', 'shape': [1, 3, 320, 320], 'dtype': 1}

Output Info:
{'name': 'output', 'shape': [1, 6300, 7], 'dtype': 1}
{'name': 'onnx::Sigmoid_369', 'shape': [1, 3, 40, 40, 7], 'dtype': 1}
{'name': 'onnx::Sigmoid_420', 'shape': [1, 3, 20, 20, 7], 'dtype': 1}
{'name': 'onnx::Sigmoid_468', 'shape': [1, 3, 10, 10, 7], 'dtype': 1}

'''
'''
Output Names: ['output', 'onnx::Sigmoid_369', 'onnx::Sigmoid_420', 'onnx::Sigmoid_468']
'''