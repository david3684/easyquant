import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader
import os
from tqdm import tqdm
import warnings
import easyquant.quantizer as quantizer
import easyquant.utils as utils

def set_weight_quantize_params(model):
    for module in model.modules():
        if isinstance(module, QModule):
            '''caculate the step size and zero point for weight quantizer'''
            module.weight_quantizer(module.weight)
            module.weight_quantizer.set_init_state(True)
              
class QModule(nn.Module):
    def __init__(self, original_module, params):
        super().__init__()
        self.original_module = original_module
        if original_module.bias is not None:
            self.bias = original_module.bias
        else:
            self.bias = None
        # Check for Conv1d, Conv2d, Conv3d
        if isinstance(original_module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            self.fwd_kwargs = dict(
                stride=original_module.stride,
                padding=original_module.padding,
                dilation=original_module.dilation,
                groups=original_module.groups
            )

            if isinstance(original_module, nn.Conv1d):
                self.fwd_func = F.conv1d
            elif isinstance(original_module, nn.Conv2d):
                self.fwd_func = F.conv2d
            elif isinstance(original_module, nn.Conv3d):
                self.fwd_func = F.conv3d
        else:
            self.fwd_kwargs = dict()
            self.fwd_func = F.linear
        self.weight_quantizer = quantizer.UniformQuantizer(params)
        #self.act_quantizer = quantizer.build_quantizer('uniform',params)
        self.weight = original_module.weight
        self.norm_function = utils.StraightThrough()
        self.activation_function = utils.StraightThrough()

    def forward(self, x):
        weight = self.weight_quantizer(self.weight)
        bias = self.bias
        out = self.fwd_func(x, weight, bias, **self.fwd_kwargs)
        out = self.norm_function(out)
        out = self.activation_function(out)
        return out
    
    def get_scale_zero_point(self):
        if self.weight_quantizer.inited:
            return self.weight_quantizer.scale, self.weight_quantizer.zero_point
        else: 
            return None, None


class QModel(nn.Module):
    """
    Make a given model into quantized model
    
    """
    def __init__(self, model, w_n_bits=8, a_n_bits=8, quantizer = 'uniform', w_optmod='minmax', a_optmod='minmax'):
        super().__init__()
        self.model = model 
        self.params = {
            'w_n_bits': w_n_bits,
            'a_n_bits': a_n_bits,
            'quantizer': quantizer,
            'w_optmod' : w_optmod,
            'a_optmod' : a_optmod
        }
        self.init_quantization(self.model)
        
    def init_quantization(self, module):
        for child_name, child_module in module.named_children():
            if isinstance(child_module, (nn.Conv2d, nn.Conv1d, nn.Conv3d, nn.Linear)): 
                quant_module = QModule(child_module, self.params)
                setattr(module, child_name, quant_module)
            else:
                self.init_quantization(child_module)
        set_weight_quantize_params(module)                                                      
    def forward(self, input):
        return self.model(input)

    
    
