import torch
import torch.nn as nn
import onnx
import numpy as np
device=torch.device("cuda")
model = torch.load('net_model.pth')  # 由研究员提供python.py文件
model.load_state_dict(torch.load('net.pth'))

# set the model to inference mode
model.eval()

x = torch.randn(57,21, 18, 64).to(device)  # 生成张量
export_onnx_file = "test.onnx"  # 目的ONNX文件名
torch.onnx.export(model,
                  x,
                  export_onnx_file,
                  opset_version=10,
                  do_constant_folding=True,  # 是否执行常量折叠优化
                  input_names=["input"],  # 输入名
                  output_names=["output"],  # 输出名
                  dynamic_axes={"input": {0: "batch_size"},  # 批处理变量
                                "output": {0: "batch_size"}})
