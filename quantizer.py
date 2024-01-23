import torch
import torch.nn as nn
import utils as utils

class UniformQuantizer(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.params = params
        self.n_levels = 2 ** self.params['w_n_bits']
        self.n_bits = self.params['w_n_bits']
        self.scale_method = self.params['w_optmod']
        self.inited = False
        self.scale = None
        self.zero_point = None
        
    def quantize(self, x):
        if self.inited == False:
            self.scale, self.zero_point = self.init_quantization_scale(x, self.scale_method)
        x_int = utils.round_ste(x / self.scale) + self.zero_point
        x_quant = torch.clamp(x_int, 0, self.n_levels - 1) #FP32
        #print(x, x_quant)
        #x_quant_int = x_quant.to(torch.int8)
        return x_quant
    
    # find initial quantization scale and zero point
    def init_quantization_scale(self, x: torch.Tensor, scale_method):
        if scale_method == 'minmax':
            print("Initializing scale with MinMax..")
            x_min = min(x.min().item(), 0)
            x_max = max(x.max().item(), 0)
            x_absmax = max(abs(x_min), x_max)
            scale = float(2*(x_absmax)) / (self.n_levels - 1)
            zero_point = round(-x_min/scale)
            print("Done")
        elif scale_method == 'mse':
            x_max = x.max()
            x_min = x.min()
            best_score = 1e+10
            for i in range(80):
                new_max = x_max * (1.0 - (i * 0.01))
                new_min = x_min * (1.0 - (i * 0.01))
                x_q = self.test_quantize(x, new_max, new_min)
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
                x_q = self.test_quantize(x, new_max, new_min)
                similarity = torch.nn.functional.cosine_similarity(x.view(-1), x_q.view(-1), dim=0)
                if similarity > best_similarity:
                    best_similarity = similarity
                    delta = (new_max - new_min) / (2 ** self.n_bits - 1)
                    zero_point = (- new_min / delta).round()
        return scale, zero_point
    
    # set inited true if scales and zero points are initialized
    def set_init_state(self, init = False):
        self.inited = init
        
    def test_quantize(self, x, max, min):
        scale = (max - min) / (2 ** self.n_bits - 1)
        zero_point = (- min / scale).round()
        x_int = torch.round(x / scale)
        x_quant = torch.clamp(x_int + zero_point, 0, self.n_levels - 1)
        x_float_q = (x_quant - zero_point) * scale
        return x_float_q
    
    
class AdaRoundLearnableQuantizer(nn.Module):
    def __init__(self, base_quantizer, weight):
        super().__init__()
        self.n_bits = base_quantizer.n_bits
        self.n_levels = base_quantizer.n_levels
        self.scale = base_quantizer.scale
        self.zero_point = base_quantizer.zero_point
        self.alpha = None # learnable alpha for adaptive rounding
        self.gamma, self.zeta = -0.1, 1.1
        self.beta = 2/3
        self.init_dequant = (weight-self.zero_point)*self.scale
        self.init_alpha(x=self.init_dequant)
        self.soft_targets = False
        self.inited = base_quantizer.inited
        
    def quantize(self, x):
        #print(self.scale, self.zero_point)
        x_floor = torch.floor(x / self.scale)
        if self.soft_targets:
            x_int = x_floor + self.get_soft_targets()
        else:
            x_int = x_floor + (self.alpha >= 0).float()
        x_quant = torch.clamp(x_int + self.zero_point, 0, self.n_levels - 1)
        x_dequant = (x_quant-self.zero_point)*self.scale
        #print(x.shape)
        #
        # print(x[0, 0, 0, 0].item(),x_int[0, 0, 0, 0].item(),x_quant[0, 0, 0, 0].item(), x_dequant[0,0,0,0].item())
        return x_quant
    
    def get_soft_targets(self):
        print(torch.sigmoid(self.alpha))
        return torch.clamp(torch.sigmoid(self.alpha) * (self.zeta - self.gamma) + self.gamma, 0, 1)
    
    def init_alpha(self, x: torch.Tensor):
        x_floor = torch.floor(x / self.scale)
        print('Init alpha to be FP32')
        rest = (x / self.scale) - x_floor  # rest of rounding [0, 1)
        alpha = -torch.log((self.zeta - self.gamma) / (rest - self.gamma) - 1)  # => sigmoid(alpha) = rest
        #print('Initial Apha :{}'.format(alpha))
        self.alpha = nn.Parameter(alpha)