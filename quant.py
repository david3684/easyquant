import torch
import torch.nn as nn
import copy
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader
import os
from tqdm import tqdm
import warnings
import utils


def init_quantization_scale(self, x: torch.Tensor, channel_wise: bool = False):
        delta=255
        zero_point=0
        ###WIP###
        return delta, zero_point
    
class QModule(nn.Module):
    def __init__(self, original_module, quantizer, weight_quant, act_quant):
        super().__init__()
        self.original_module = copy.deepcopy.original_module
        self.weight_quant = weight_quant
        self.act_quant = act_quant
        self.weight = original_module.weight

    def forward(self, x):
        if self.weight_quant:
            weight = self.quantize_weights(self.weight)
        return self.original_module(x)
    
    def quantize_weights(self, weights):
        return 
    def quantize_activation(self, x):
        return

class QModel:
    """
    Make a given model into quantized model
    
    :param model: Loaded Model.
    :param n_bits: Quantization Bitwidth.
    :param symmetric: Transformations to be applied to the dataset samples.
    :param weight_quant: if True, weights are quantized.
    :param act_quant: if True, activations are quantized.
    :param quantizer: Quantizer choice. Uniform or AdaRound or Smooth
    """
    def __init__(self, model, n_bits, weight_quant = True, act_quant = False, quantizer = 'Uniform'):
    
        self.model = model
        self.n_bits = n_bits
        self.quantizer = quantizer
        self.weight_quant = weight_quant
        self.act_quant = act_quant
        

    def _apply_quantization(self, module):
        for child_name, child_module in module.named_children():
            if isinstance(child_module, (nn.Conv2d, nn.Linear)):
                quant_module = QModule(child_module, self.quantizer, self.weight_quant, self.act_quant)
                setattr(module, child_name, quant_module)
            else:
                self._apply_quantization(child_module) 
                                                     
            
    def quantize(self):
        self._apply_quantization(self.model)
        return self.model

    #def quantize_layer(self, layer):
    #    weight = layer.weight.data
    #    print("Original Weights: {}".format(weight))

        
    #    delta, zero_point = self.init_quantization_scale(weight, self.channel_wise)

        
    #    quantized_weight = torch.round((weight - zero_point) / delta).int()
    #    print("Quantized Weights: {}".format(quantized_weight))

        
    #    dequantized_weight = (quantized_weight.float() * delta) + zero_point
    #    print("DeQuantized Weights: {}".format(dequantized_weight))

    #    quantized_layer = copy.deepcopy(layer)
    #    quantized_layer.weight.data = dequantized_weight

    #    return quantized_layer
    
    
    
