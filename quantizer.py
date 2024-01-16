import torch
import torch.nn as nn
import easyquant.utils as utils

class UniformQuantizer(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.params = params
        self.num_levels = 2 ** self.params['w_n_bits']
        self.scale_method = self.params['w_optmod']
        
    def forward(self, x):
        self.scale, self.zero_point = self.init_quantization_scale(x, self.scale_method)
        x_int = utils.round_ste(x / self.scale) + self.zero_point
        x_quant = torch.clamp(x_int, 0, self.num_levels - 1)
        print('float:{}, quantized weight:{}'.format(x, x_quant))
        # x_dequant = (x_quant - self.zero_point) * self.scale
        return x_quant
    
    # find initial quantization scale and zero point
    def init_quantization_scale(self, x: torch.Tensor, scale_method):
        if scale_method == 'minmax':
            x_min = min(x.min().item(), 0)
            x_max = max(x.max().item(), 0)
            x_absmax = max(abs(x_min), x_max)
            scale = float(2*(x_absmax)) / (self.num_levels - 1)
            zero_point = round(-x_min/scale)
        elif scale_method == 'mse':
            x_max = x.max()
            x_min = x.min()
            best_score = 1e+10
            for i in range(80):
                new_max = x_max * (1.0 - (i * 0.01))
                new_min = x_min * (1.0 - (i * 0.01))
                x_q = self.quantize(x, new_max, new_min)
                score = utils.lp_loss(x, x_q, p=2.4, reduction='all')
                if score < best_score:
                    best_score = score
                    scale = (new_max - new_min) / (2 ** self.n_bits - 1)
                    zero_point = (- new_min / scale).round()
        elif scale_method == 'cosine':
            x_max = x.max()
            x_min = x.min()
            best_similarity = -1
            for i in range(80):
                new_max = x_max * (1.0 - (i * 0.01))
                new_min = x_min * (1.0 - (i * 0.01))
                x_q = self.quantize(x, new_max, new_min)
                # Cosine similarity 계산
                similarity = torch.nn.functional.cosine_similarity(x.view(-1), x_q.view(-1), dim=0)
                if similarity > best_similarity:
                    best_similarity = similarity
                    delta = (new_max - new_min) / (2 ** self.n_bits - 1)
                    zero_point = (- new_min / delta).round()
        return scale, zero_point
    
    def quantize(self, x, max, min):
        scale = (max - min) / (2 ** self.n_bits - 1)
        zero_point = (- min / scale).round()
        x_int = torch.round(x / scale)
        x_quant = torch.clamp(x_int + zero_point, 0, self.n_levels - 1)
        x_float_q = (x_quant - zero_point) * scale
        return x_float_q

class AdaroundQuantizer(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.params = params
    def forward(self, weights):
        # Adaround quantization logic
        # ...
        return
class build_quantizer(nn.Module):
    def __init__(self, type, params):
        super().__init__()
        self.type = type
        if self.type == "uniform":
            self.quantizer = UniformQuantizer(params)
        elif self.type == "adaround":
            self.quantizer = AdaroundQuantizer(params)
    def forward(self, x):
        return self.quantizer(x)
        
    