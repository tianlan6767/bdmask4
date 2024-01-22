################################################################################
# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
################################################################################
import os
import os.path as osp
import re
from typing import List, Callable, Union, Dict
from tqdm import tqdm
from copy import deepcopy

# PyTorch
import torch
import torch.optim as optim
from torch.cuda import amp
import torch.nn.functional as F

# Pytorch Quantization
from pytorch_quantization import nn as quant_nn
from pytorch_quantization.nn.modules import _utils as quant_nn_utils
from pytorch_quantization import calib
from pytorch_quantization.tensor_quant import QuantDescriptor
from pytorch_quantization import quant_modules
from pytorch_quantization import tensor_quant
from absl import logging as quant_logging
from detectron2.modeling.backbone.resnet import BasicBlock



# Custom Rules
from quantization.rules import find_quantizer_pairs

import time
def time_synchronized():
    torch.cuda.synchronize()
    return time.time()

class QuantAdd(torch.nn.Module):
    def __init__(self, quantization):
        super().__init__()

        if quantization:
            self._input0_quantizer = quant_nn.TensorQuantizer(QuantDescriptor(num_bits=8, calib_method="histogram"))
            # self._input1_quantizer = quant_nn.TensorQuantizer(QuantDescriptor(num_bits=8, calib_method="histogram"))
            self._input0_quantizer._calibrator._torch_hist = True
            # self._input1_quantizer._calibrator._torch_hist = True
            self._fake_quant = True
        self.quantization = quantization

    def forward(self, x, y):
        if self.quantization:
            # print(f"QAdd {self._input0_quantizer}  {self._input1_quantizer}")
            # return self._input0_quantizer(x) + self._input1_quantizer(y)
            return x + self._input0_quantizer(y)
        return x + y

class QuantBasicBlock(BasicBlock):
    def __init__(self, in_channels, out_channels,*, stride=1, norm="BN", quantize_residual=True):
        super().__init__(in_channels, out_channels, stride=stride, norm=norm)
        self._quantize_residual = quantize_residual
        if self._quantize_residual:
            self.residual_quantizer = quant_nn.TensorQuantizer(
                quant_nn.QuantConv2d.default_quant_desc_input)

    def forward(self, x):
        out = self.conv1(x)
        out = F.relu_(out)
        out = self.conv2(out)

        if self.shortcut is not None:
            shortcut = self.shortcut(x)
        else:
            shortcut = x
            
        if self._quantize_residual:
            out += self.residual_quantizer(shortcut)
        else:
            out += shortcut

        out = F.relu_(out)
        return out
    
    

class disable_quantization:
    def __init__(self, model):
        self.model  = model

    def apply(self, disabled=True):
        for name, module in self.model.named_modules():
            if isinstance(module, quant_nn.TensorQuantizer):
                module._disabled = disabled

    def __enter__(self):
        self.apply(True)

    def __exit__(self, *args, **kwargs):
        self.apply(False)


class enable_quantization:
    def __init__(self, model):
        self.model  = model

    def apply(self, enabled=True):
        for name, module in self.model.named_modules():
            if isinstance(module, quant_nn.TensorQuantizer):
                module._disabled = not enabled

    def __enter__(self):
        self.apply(True)
        return self

    def __exit__(self, *args, **kwargs):
        self.apply(False)


def have_quantizer(module):
    for name, module in module.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            return True


# Initialize PyTorch Quantization
def initialize():
    quant_desc_input = QuantDescriptor(calib_method="histogram")
    quant_nn.QuantConv2d.set_default_quant_desc_input(quant_desc_input)
    quant_nn.QuantMaxPool2d.set_default_quant_desc_input(quant_desc_input)
    quant_nn.QuantLinear.set_default_quant_desc_input(quant_desc_input)
    quant_logging.set_verbosity(quant_logging.ERROR)


def transfer_torch_to_quantization(nninstance : torch.nn.Module, quantmodule):

    quant_instance = quantmodule.__new__(quantmodule)
    for k, val in vars(nninstance).items():
        setattr(quant_instance, k, val)

    def __init__(self):
        quant_desc_input, quant_desc_weight = quant_nn_utils.pop_quant_desc_in_kwargs(self.__class__)
        if isinstance(self, quant_nn_utils.QuantInputMixin):
            self.init_quantizer(quant_desc_input)

            # Turn on torch_hist to enable higher calibration speeds
            if isinstance(self._input_quantizer._calibrator, calib.HistogramCalibrator):
                self._input_quantizer._calibrator._torch_hist = True
        else:
            self.init_quantizer(quant_desc_input, quant_desc_weight)

            # Turn on torch_hist to enable higher calibration speeds
            if isinstance(self._input_quantizer._calibrator, calib.HistogramCalibrator):
                self._input_quantizer._calibrator._torch_hist = True
                self._weight_quantizer._calibrator._torch_hist = True

    __init__(quant_instance)
    return quant_instance


def quantization_ignore_match(ignore_policy : Union[str, List[str], Callable], path : str) -> bool:

    if ignore_policy is None: return False
    if isinstance(ignore_policy, Callable):
        return ignore_policy(path)

    if isinstance(ignore_policy, str) or isinstance(ignore_policy, List):

        if isinstance(ignore_policy, str):
            ignore_policy = [ignore_policy]

        if path in ignore_policy: return True
        for item in ignore_policy:
            if re.match(item, path):
                return True
    return False


# For example: YoloV5 Bottleneck
def bottleneck_quant_forward(self, x):
    if hasattr(self, "addop"):
        return self.addop(x, self.cv2(self.cv1(x))) if self.add else self.cv2(self.cv1(x))
    return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

def basicblock_quant_forward(self, x):
    out = self.conv1(x)
    out = F.relu_(out)
    out = self.conv2(out)
    if self.shortcut is not None:
        shortcut = self.shortcut(x)
    else:
        shortcut = x
    if hasattr(self, "addop"):
    # out += self.residual_quantizer(shortcut)
        return F.relu_(self.addop(out, shortcut))
    out += shortcut
    return F.relu_(out)


# For example: YoloV5 Bottleneck
def replace_bottleneck_forward(model):
    for name, bottleneck in model.named_modules():
        if bottleneck.__class__.__name__ == "Bottleneck":
            pass
            # if bottleneck.add:
            #     if not hasattr(bottleneck, "addop"):
            #         print(f"Add QuantAdd to {name}")
            #         bottleneck.addop = QuantAdd(bottleneck.add)
            #     bottleneck.__class__.forward = bottleneck_quant_forward
        elif bottleneck.__class__.__name__ == "BasicBlock":            
            if not hasattr(bottleneck, "addop"):
                    print(f"Add QuantAdd to {name}")
                    # bottleneck = QuantBasicBlock(in_channels=bottleneck.in_channels,
                    #                              out_channels=bottleneck.out_channels, 
                    #                              stride=bottleneck.stride,norm=bottleneck.norm, quantize_residual=True)
                    # bottleneck.__class__.forward = basicblock_quant_forward
                    bottleneck.addop = QuantAdd(True) 
                    bottleneck.__class__.forward = basicblock_quant_forward
                    


def replace_to_quantization_module(model : torch.nn.Module, ignore_policy : Union[str, List[str], Callable] = None):

    module_dict = {}
    for entry in quant_modules._DEFAULT_QUANT_MAP:
        module = getattr(entry.orig_mod, entry.mod_name)
        module_dict[id(module)] = entry.replace_mod
    module_dict[id(type(model.backbone.fpn_lateral3))] = module_dict[id(torch.nn.modules.conv.Conv2d)]
    def recursive_and_replace_module(module, prefix=""):
        for name in module._modules:
            submodule = module._modules[name]
            path      = name if prefix == "" else prefix + "." + name
            recursive_and_replace_module(submodule, path)

            submodule_id = id(type(submodule))
            if submodule_id in module_dict:
                ignored = quantization_ignore_match(ignore_policy, path)
                if ignored:
                    print(f"Quantization: {path} has ignored.")
                    continue
                    
                module._modules[name] = transfer_torch_to_quantization(submodule, module_dict[submodule_id])

    recursive_and_replace_module(model)

def calibrate_model(cfg, model : torch.nn.Module,  dataloader,  device, num_batch=10000):

    def compute_amax(model, **kwargs):
        for name, module in model.named_modules():
            if isinstance(module, quant_nn.TensorQuantizer):
                if module._calibrator is not None:
                    if isinstance(module._calibrator, calib.MaxCalibrator):
                        module.load_calib_amax(strict=False)
                    else:
                        module.load_calib_amax(**kwargs)

                    module._amax = module._amax.to(device)
        
    def collect_stats(cfg, model, data_loader, device, num_batch=200):
        """Feed data to the network and collect statistics"""
        # Enable calibrators        
        model.eval()
        for name, module in model.named_modules():
            if isinstance(module, quant_nn.TensorQuantizer):
                if module._calibrator is not None:
                    module.disable_quant()
                    module.enable_calib()
                else:
                    module.disable()

        # Feed data to the network for collecting stats
        with torch.no_grad():
            for i, datas in tqdm(enumerate(data_loader), total=min(len(data_loader), num_batch), desc="Collect stats for calibrating"):
                inf_t0 = time_synchronized()
                model(datas)
                inf_t1 = time_synchronized()
                print("当前耗时推理耗时:", inf_t1 - inf_t0)
                if i >= num_batch:
                    break

        # Disable calibrators
        for name, module in model.named_modules():
            if isinstance(module, quant_nn.TensorQuantizer):
                if module._calibrator is not None:
                    module.enable_quant()
                    module.disable_calib()
                else:
                    module.enable()
    t0 = time_synchronized()
    collect_stats(cfg, model, dataloader, device, num_batch=num_batch)
    t1 = time_synchronized()
    compute_amax(model, method="mse", strict=False)
    t2 = time_synchronized()
    print("当前耗时:", t2 - t1, t1 - t0)


def finetune(
    model : torch.nn.Module, train_dataloader, per_epoch_callback : Callable = None, preprocess : Callable = None,
    nepochs=10, early_exit_batchs_per_epoch=1000, lrschedule : Dict = None, fp16=True, learningrate=1e-5,
    supervision_policy : Callable = None
):
    origin_model = deepcopy(model).eval()
    disable_quantization(origin_model).apply()

    model.train()
    model.requires_grad_(True)

    scaler       = amp.GradScaler(enabled=fp16)
    optimizer    = optim.Adam(model.parameters(), learningrate)
    quant_lossfn = torch.nn.MSELoss()
    device       = next(model.parameters()).device

    if lrschedule is None:
        lrschedule = {
            0: 1e-6,
            3: 1e-5,
            8: 1e-6
        }

    def make_layer_forward_hook(l):
        def forward_hook(m, input, output):
            l.append(output)
        return forward_hook

    supervision_module_pairs = []
    for ((mname, ml), (oriname, ori)) in zip(model.named_modules(), origin_model.named_modules()):
        if isinstance(ml, quant_nn.TensorQuantizer): continue

        if supervision_policy:
            if not supervision_policy(mname, ml):
                continue

        supervision_module_pairs.append([ml, ori])


    for iepoch in range(nepochs):

        if iepoch in lrschedule:
            learningrate = lrschedule[iepoch]
            for g in optimizer.param_groups:
                g["lr"] = learningrate

        model_outputs  = []
        origin_outputs = []
        remove_handle  = []

        for ml, ori in supervision_module_pairs:
            remove_handle.append(ml.register_forward_hook(make_layer_forward_hook(model_outputs))) 
            remove_handle.append(ori.register_forward_hook(make_layer_forward_hook(origin_outputs)))

        model.train()
        pbar = tqdm(train_dataloader, desc="QAT", total=early_exit_batchs_per_epoch)
        for ibatch, imgs in enumerate(pbar):

            if ibatch >= early_exit_batchs_per_epoch:
                break
            
            if preprocess:
                imgs = preprocess(imgs)
                
            imgs = imgs.to(device)
            with amp.autocast(enabled=fp16):
                model(imgs)

                with torch.no_grad():
                    origin_model(imgs)

                quant_loss = 0
                for index, (mo, fo) in enumerate(zip(model_outputs, origin_outputs)):
                    quant_loss += quant_lossfn(mo, fo)

                model_outputs.clear()
                origin_outputs.clear()

            if fp16:
                scaler.scale(quant_loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                quant_loss.backward()
                optimizer.step()
            optimizer.zero_grad()
            pbar.set_description(f"QAT Finetuning {iepoch + 1} / {nepochs}, Loss: {quant_loss.detach().item():.5f}, LR: {learningrate:g}")

        # You must remove hooks during onnx export or torch.save
        for rm in remove_handle:
            rm.remove()

        if per_epoch_callback:
            if per_epoch_callback(model, iepoch, learningrate):
                break


def export_onnx(model, input, file, *args, **kwargs):

    quant_nn.TensorQuantizer.use_fb_fake_quant = True

    model.eval()
    with torch.no_grad():
        torch.onnx.export(model, input, file, *args, **kwargs)

    quant_nn.TensorQuantizer.use_fb_fake_quant = False
