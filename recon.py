import quantizer, utils, torch
import torch.nn.functional as F
import torch.optim
from tqdm import tqdm
from quant import QModule, QModel
from quantizer import AdaRoundLearnableQuantizer, UniformQuantizer

def reconstruct(qmodel, fpmodel, calibration_set, reconstruction_method='layer', loss_type='mse', iters=100):
    """
    Reconstruct the quantized model using the calibration set.
    """
    fp_modules = dict(fpmodel.named_modules())

    for name, qmodule in qmodel.named_modules():
        if isinstance(qmodule, QModule):
            # 'model.' 접두사 제거
            fp_module_name = name.replace('model.', '')

            if fp_module_name in fp_modules:
                fp_module = fp_modules[fp_module_name]
                print(f"Reconstructing layer: {name}")

                if reconstruction_method == 'layer':
                    layer_reconstruction(qmodel, fpmodel, qmodule, fp_module, calibration_set, loss_type, iters)
                else:
                    raise NotImplementedError(f"Reconstruction method '{reconstruction_method}' not implemented")
            else:
                print(f"FP module for {name} not found")
    print("Reconstruction completed.")
    
class LayerOutputHook:
    def __init__(self, layer):
        self.hook = layer.register_forward_hook(self.hook_fn)
        self.input = None
        self.output = None

    def hook_fn(self, module, x, output):
        self.input = x[0]
        self.output = output

    def remove(self):
        self.hook.remove()
        
def layer_reconstruction(qmodel, fpmodel, layer, fpmodule, cali_set, loss_type, iters):
    
    
    loss_func = Loss(layer, p=2, loss_type=loss_type)
    layer.weight_quantizer = AdaRoundLearnableQuantizer(base_quantizer=layer.weight_quantizer, weight = layer.quantized_weight)
    layer.weight_quantizer.soft_targets = True
    w_opttarget = [layer.weight_quantizer.alpha]
    lr = 3e-3
    optimizer = torch.optim.Adam(w_opttarget, lr=lr)
    layer.reconstructing = True
    fp_output_hook = LayerOutputHook(fpmodule)
    q_output_hook = LayerOutputHook(layer)
    t = tqdm(range(iters), desc='Reconstructing layer')
    
    for i in t:
        for data, _ in cali_set:
            data = data.cuda()
            optimizer.zero_grad()
            _ = fpmodel(data)
            target_output = fp_output_hook.output 
            _ = qmodel(data)
            quant_output = q_output_hook.output
            print(target_output, quant_output)
            loss = loss_func(quant_output, target_output)
            loss.backward(retain_graph=True)
            optimizer.step()
            print(loss.item())
            t.set_postfix(loss=loss.item())

    fp_output_hook.remove()
    q_output_hook.remove()
    layer.reconstructing = False
    layer.weight_quantizer.soft_targets = False

class Loss:
    def __init__(self, layer, p, loss_type):
        self.layer = layer
        self.p = p
        self.loss_type = loss_type
        
    def __call__(self, pred, target):
        if self.loss_type == 'mse':
            rec_loss = utils.lp_loss(pred, target, p=self.p)  # p=2
        elif self.loss_type == 'cosine':
            similarity = F.cosine_similarity(pred.view(-1), target.view(-1), dim=0)
            rec_loss = 1 - similarity.mean()
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")
        return rec_loss