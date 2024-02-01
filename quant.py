import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader
import os
from tqdm import tqdm
import warnings
import quantizer as quantizer
import utils as utils

def set_weight_quantize_params(module, method):
    """
    Check if given module is qmodule, and apply initial quantization and save it, set init state true
    """
    if isinstance(module, QModule):
        print("Initializing {} with {}".format(module.module_path, method))
        module.quantized_weight = module.weight_quantizer.quantize(module.origin_weight) 
        module.weight_quantizer.set_init_state(True)
class QModule(nn.Module):
    """
    Main module that performs quantization for convolution and linear layers
    """
    def __init__(self, original_module, module_path, params):
        super().__init__()
        self.original_module = original_module
        self.module_path = module_path
        self.params = params
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
        self.weight_quantizer = quantizer.UniformQuantizer(params, True)
        if self.params['a_n_bits'] is not None:
            self.act_quantizer = quantizer.UniformQuantizer(params,False)
        self.origin_weight = original_module.weight.data.clone()
        self.quantized_weight = None
        self.norm_function = utils.StraightThrough()
        self.activation_function = utils.StraightThrough()
        self.reconstructing = False
        self.calibrated = False
        self.cached_weight = None #cached weight used for final output

    def forward(self, x):
        """
        
        """
        if self.reconstructing:
            self.quantized_weight = self.weight_quantizer.quantize(self.origin_weight) # requantize weight with AdaRound quantizer
        scale, zero_point = self.get_scale_zero_point()
        weight = (self.quantized_weight - zero_point) * scale
        bias = self.bias
        out = self.fwd_func(x, weight, bias, **self.fwd_kwargs)
        out = self.norm_function(out)
        out = self.activation_function(out)
        if self.params['a_n_bits'] is not None:
            out = self.act_quantizer.quantize(out)
            out = (out - self.act_quantizer.zero_point) * self.act_quantizer.scale
        return out
    
    def get_scale_zero_point(self):
        if self.weight_quantizer.inited:
            return self.weight_quantizer.scale, self.weight_quantizer.zero_point
        else: 
            return None, None


class QModel(nn.Module):
    """
    Make a given model into quantized model
    :param w_n_bits: number of bit for weight quantization
    :param init_method: method for initializing scale and zero point for weight quantization
    :param w_optmod: loss for reconstructing weight quantization paramters
    """
    def __init__(self, model, w_n_bits=8, a_n_bits=None, init_method = 'minmax'):
        super().__init__()
        self.model = model 
        self.params = {
            'w_n_bits': w_n_bits,
            'a_n_bits': a_n_bits,
            'init_method': init_method,
        }
        self.init_quantization(self.model)
        print("Complete initializing QModel")
        
    def init_quantization(self, module, parent_name=''):
        """
        Initialize quantization. Change convolution and linear modules into qmodule, and set weight quantization parameters for them.
        """
        for child_name, child_module in module.named_children():
            module_path = parent_name + '.' + child_name if parent_name else child_name
            if isinstance(child_module, (nn.Conv2d, nn.Conv1d, nn.Conv3d, nn.Linear)): 
                quant_module = QModule(child_module, module_path, self.params)
                setattr(module, child_name, quant_module)
                set_weight_quantize_params(quant_module, self.params['init_method'])
            else:
                self.init_quantization(child_module, module_path)
    def forward(self, input):
        return self.model(input)

    
    
