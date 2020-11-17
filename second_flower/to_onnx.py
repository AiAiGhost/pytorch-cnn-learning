import torch
import torch.nn as nn
import settings


file_name = "g_3_3x3_6_bn_res_x3/cnn_20200919_141805"
model_file_path = settings.get_log_path() + file_name + '.pkl'
onnx_file_path = settings.get_log_path() + file_name + '.onnx'

device = settings.get_device()


model = torch.load(model_file_path)
model = model.to(device)
print(model)

input = torch.randn(1, 3, 224, 224, device = device)
torch.onnx.export(model, input, onnx_file_path, verbose=True)

