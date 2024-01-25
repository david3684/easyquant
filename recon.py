import quantizer, utils, torch
import torch.nn.functional as F
import torch.optim
from tqdm import tqdm
from quant import QModule, QModel
from quantizer import AdaRoundLearnableQuantizer, UniformQuantizer

def reconstruct(qmodel, fpmodel, calibration_set, adaround = True, reconstruction_method='layer', loss_type='mse',  iters=3):
    """
    Reconstruct the quantized model using the calibration set.
    """
    fp_modules = dict(fpmodel.named_modules())

    for name, qmodule in qmodel.named_modules():
        if isinstance(qmodule, QModule):
            if 'downsample' in name:
                print(f"Skipping downsampling layer: {name}")
                continue
            fp_module_name = name.replace('model.', '')
            if fp_module_name in fp_modules:
                fp_module = fp_modules[fp_module_name]
                #print(qmodule, fp_module)
                print(f"Reconstructing layer: {name}")

                if reconstruction_method == 'layer':
                    layer_reconstruction(qmodel, fpmodel, qmodule, fp_module, calibration_set, loss_type, iters, adaround)
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


def layer_reconstruction(qmodel, fpmodel, layer, fp_layer, cali_set, loss_type, iters, adaround = True):
    print('Start Caching')
    cached_q_inputs = get_input(qmodel, layer, cali_set, keep_gpu=False)
    cached_fp_outputs = get_output(fpmodel, fp_layer, cali_set, keep_gpu=False)
    if adaround:
        loss_func = Loss(layer, round_loss='relaxation', weight=0.001,
                                max_count=iters, rec_loss='mse', b_range=(20,2),
                                decay_start=0, warmup=0.0, p=2)
        layer.weight_quantizer = AdaRoundLearnableQuantizer(base_quantizer=layer.weight_quantizer, weight= layer.origin_weight)
        layer.weight_quantizer.soft_targets = True
        w_opttarget = [layer.weight_quantizer.alpha]
    else:
        loss_func = Loss(layer, round_loss=None, weight=0.001,
                                max_count=iters, rec_loss='mse', b_range=(20,2),
                                decay_start=0, warmup=0.0, p=2)
        w_opttarget = [layer.weight_quantizer.scale]
    lr = 5e-3
    optimizer = torch.optim.Adam(w_opttarget, lr=lr)
    layer.reconstructing = True
    
    batch_size = 32
    
    print('Start Iteration')
    t = tqdm(range(iters))
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


class LinearTempDecay:
    def __init__(self, t_max: int, rel_start_decay: float = 0.2, start_b: int = 10, end_b: int = 2):
        self.t_max = t_max
        self.start_decay = rel_start_decay * t_max
        self.start_b = start_b
        self.end_b = end_b

    def __call__(self, t):
        """
        Cosine annealing scheduler for temperature b.
        :param t: the current time step
        :return: scheduled temperature
        """
        if t < self.start_decay:
            return self.start_b
        else:
            rel_t = (t - self.start_decay) / (self.t_max - self.start_decay)
            return self.end_b + (self.start_b - self.end_b) * max(0.0, (1 - rel_t))
        
class Loss:
    def __init__(self,
                 layer: QModule,
                 round_loss: str = 'relaxation',
                 weight: float = 1.,
                 rec_loss: str = 'mse',
                 max_count: int = 2000,
                 b_range: tuple = (10, 2),
                 decay_start: float = 0.0,
                 warmup: float = 0.0,
                 p: float = 2.):

        self.layer = layer
        self.round_loss = round_loss
        self.weight = weight
        self.rec_loss = rec_loss
        self.loss_start = max_count * warmup
        self.p = p

        self.temp_decay = LinearTempDecay(max_count, rel_start_decay=warmup + (1 - warmup) * decay_start,
                                          start_b=b_range[0], end_b=b_range[1])
        self.count = 0

    def __call__(self, pred, tgt, grad=None):
        """
        Compute the total loss for adaptive rounding:
        rec_loss is the quadratic output reconstruction loss, round_loss is
        a regularization term to optimize the rounding policy

        :param pred: output from quantized model
        :param tgt: output from FP model
        :param grad: gradients to compute fisher information
        :return: total loss function
        """
        self.count += 1
        if self.rec_loss == 'mse':
            rec_loss = utils.lp_loss(pred, tgt, p=self.p)
        elif self.rec_loss == 'fisher_diag':
            rec_loss = ((pred - tgt).pow(2) * grad.pow(2)).sum(1).mean()
        elif self.rec_loss == 'fisher_full':
            a = (pred - tgt).abs()
            grad = grad.abs()
            batch_dotprod = torch.sum(a * grad, (1, 2, 3)).view(-1, 1, 1, 1)
            rec_loss = (batch_dotprod * a * grad).mean() / 100
        else:
            raise ValueError('Not supported reconstruction loss function: {}'.format(self.rec_loss))

        b = self.temp_decay(self.count)
        #print(self.layer.weight_quantizer.alpha)
        if self.count < self.loss_start or self.round_loss == 'none':
            b = round_loss = 0
        elif self.round_loss == 'relaxation':
            round_loss = 0
            round_vals = self.layer.weight_quantizer.get_soft_targets()
            #print(round_vals)
            round_loss += self.weight * (1 - ((round_vals - .5).abs() * 2).pow(b)).sum()
        else:
            round_loss = 0

        total_loss = rec_loss + round_loss
        if self.count % 2000 == 0:
            print('Total loss:\t{:.3f} (rec:{:.3f}, round:{:.3f})\tb={:.2f}\tcount={}'.format(
                  float(total_loss), float(rec_loss), float(round_loss), b, self.count))
        return total_loss