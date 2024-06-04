import torch
import torch.nn as nn
import json
from torch.onnx.symbolic_helper import _parse_arg

# 这里有时间换空间的想法
class SwishAndBiasImplementation(torch.autograd.Function):

    @staticmethod
    def symbolic(g, input, bias):
        # name_s，在g.op这个场景下，有两个作用
        # 1. 告诉g.op，这个是属性
        # 2. 告诉g.op，这个属性的名称是name，类型是s，也就是string
        #
        # 老师提供的tensorRT框架插件做了定义
        # 1. 名称必须是Plugin
        # 2. name_s指定插件的子名称
        # 3. info_s指定需要带进去的属性信息
        #
        # 如果按照官方的写法，那么插件名称可以任意，属性可以任意
        # 即不需要一定写name_s，可以任意写
        return g.op("Plugin", input, bias, name_s="Swish", info_s=json.dumps({
            "size": 555,
            "shape": [6, 7, 8],
            "module":  "abcdefg"
        }))

    @staticmethod
    def forward(context, input, bias):
        # 告诉pytorch，这个input需要保留到反向时使用
        # 如果不告诉，则input会在节点forward后内存被回收复用
        # 这是因为内存高效复用的一个原则
        context.save_for_backward(input)

        # y = x * torch.sigmoid(x)
        # x -> sigmoid -> x'
        # x * x' -> y
        return input * torch.sigmoid(input) + bias

    @staticmethod
    def backward(context, grad_output):
        input = context.saved_tensors[0]
        sigmoid_input = torch.sigmoid(input)

        # 返回的第一个tensor，是input的导数
        # 返回的第二个tensor，是bias的导数
        return grad_output * (sigmoid_input * (1 + input * (1 - sigmoid_input))), grad_output


class MemoryEfficientSwish(nn.Module):
    def __init__(self):
        super().__init__()

        # 创建一个参数
        self.bias = nn.Parameter(torch.full((1,), 3.15))
    
    def forward(self, x):
        return SwishAndBiasImplementation.apply(x, self.bias)


class Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv = nn.Conv2d(1, 1, 3, stride=1, padding=0, bias=False)
        self.conv.weight.data = torch.FloatTensor([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ]).view(1, 1, 3, 3)  # output, input, height, width
        self.swish = MemoryEfficientSwish()

    def forward(self, x):
        return self.swish(self.conv(x))


input = torch.arange(9).view(1, 1, 3, 3).float()
print(input)

model = Model()
model.train()

y = model(input)
loss = y.mean()
loss.backward()

print(loss)
print(model.swish.bias.grad)


import torch.onnx
model.eval()

torch.onnx.export(model, (input,), 
    "/media/ps/data1/train/LQ/task/bdm/bdmask/workspace/code/plugin/weights/plugin.onnx", 
    opset_version=11,
    operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK,
    verbose=True)
    