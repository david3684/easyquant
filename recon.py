import quantizer, utils, torch
import torch.nn.functional as F
import torch.optim
from tqdm import tqdm
from quant import QModule, QModel
from quantizer import AdaRoundLearnableQuantizer, UniformQuantizer

def reconstruct(qmodel, fpmodel, calibration_set, reconstruction_method='layer', loss_type='mse', iters=1000):
    """
    Reconstruct the quantized model using the calibration set.
    """
    fp_modules = dict(fpmodel.named_modules())

    for name, qmodule in qmodel.named_modules():
        if isinstance(qmodule, QModule):
            
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
    
def get_input(model, layer, loader, batch_size=32, keep_gpu=True):
    """
    Get input data for a particular layer over calibration dataset using DataLoader.

    :param model: Model for which inputs are to be cached.
    :param layer: Layer for which inputs are to be cached.
    :param loader: DataLoader for the calibration dataset.
    :param batch_size: Batch size for processing.
    :param keep_gpu: Keep data on GPU if True, else move to CPU.
    :return: Cached input data for the specified layer.
    """
    device = next(model.parameters()).device
    cached_inputs = []
    model.eval()

    # Hook for extracting the inputs to the layer
    def hook_fn(module, inp, out):
        cached_inputs.append(inp[0].detach().cpu())

    hook = layer.register_forward_hook(hook_fn)

    with torch.no_grad():
        for data, _ in loader:
            data = data.to(device)
            model(data)
            torch.cuda.empty_cache()

            if len(cached_inputs) * batch_size >= loader.dataset.__len__():
                break

    hook.remove()

    cached_inps = torch.cat(cached_inputs, dim=0)

    if not keep_gpu:
        cached_inps = cached_inps.cpu()

    torch.cuda.empty_cache()

    return cached_inps  

def get_output(model, layer, loader, batch_size=32, keep_gpu=True):
    """
    Get output data for a particular layer over calibration dataset using DataLoader.

    :param model: Model for which outputs are to be cached.
    :param layer: Layer for which outputs are to be cached.
    :param loader: DataLoader for the calibration dataset.
    :param batch_size: Batch size for processing.
    :param keep_gpu: Keep data on GPU if True, else move to CPU.
    :return: Cached output data for the specified layer.
    """
    device = next(model.parameters()).device
    cached_outputs = []
    model.eval()

    # Hook for extracting the outputs from the layer
    hook = layer.register_forward_hook(lambda mod, inp, out: cached_outputs.append(out.detach().cpu()))

    with torch.no_grad():
        for data, _ in loader:
            data = data.to(device)
            model(data)
            torch.cuda.empty_cache()

            if len(cached_outputs) * batch_size >= loader.dataset.__len__():
                break

    hook.remove()

    cached_outs = torch.cat(cached_outputs, dim=0)
    if not keep_gpu:
        cached_outs = cached_outs.cpu()

    torch.cuda.empty_cache()

    return cached_outs


def layer_reconstruction(qmodel, fpmodel, layer, fp_layer, cali_set, loss_type, iters):
    
    cached_q_inputs = get_input(qmodel, layer, cali_set, keep_gpu=False)
    cached_fp_outputs = get_output(fpmodel, fp_layer, cali_set, keep_gpu=False)
    loss_func = Loss(layer, p=2, loss_type=loss_type)
    layer.weight_quantizer = AdaRoundLearnableQuantizer(base_quantizer=layer.weight_quantizer, weight = layer.quantized_weight)
    layer.weight_quantizer.soft_targets = True
    w_opttarget = [layer.weight_quantizer.alpha]
    lr = 3e-3
    optimizer = torch.optim.Adam(w_opttarget, lr=lr)
    layer.reconstructing = True
    
    batch_size = 32
    
    
    t = tqdm(range(iters), desc='Reconstructing layer')
    for i in t:
        indices = torch.randint(0, len(cached_q_inputs), (batch_size,))

        optimizer.zero_grad()
        target_output = torch.index_select(cached_fp_outputs, 0, indices).to('cuda')
        quant_inputs = torch.index_select(cached_q_inputs, 0, indices).to('cuda')
        quant_output = layer(quant_inputs)
        
        loss = loss_func(quant_output, target_output)
        loss.backward(retain_graph=True)
        optimizer.step()
        t.set_postfix(loss=loss.item())

    layer.reconstructing = False
    layer.weight_quantizer.soft_targets = False

class Loss:
    def __init__(self, layer, p, loss_type):
        self.layer = layer
        self.p = p
        self.loss_type = loss_type
        self.count = 0
        
    def __call__(self, pred, target):
        self.count += 1
        
        rec_loss = utils.lp_loss(pred, target, p=self.p)
        
        round_loss = 0
        pd_loss = 0
        total_loss = rec_loss + round_loss + pd_loss
        
        return total_loss