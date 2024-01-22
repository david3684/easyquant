import quantizer, utils, torch
import torch.nn.functional as F
from quant import QModule, QModel
def reconstruct(model, calibration_set, reconstruction_method='layer', loss_type='mse', iters=100):
    """
    Reconstruct the quantized model using the calibration set.
    """
    for name, module in model.named_modules():
        if isinstance(module, QModule):
            print(f"Reconstructing layer: {name}")
            if reconstruction_method == 'layer':
                layer_reconstruction(model, module, calibration_set, loss_type, iters)
            else:
                raise NotImplementedError(f"Reconstruction method '{reconstruction_method}' not implemented")
    print("Reconstruction completed.")
    
    
def layer_reconstruction(model, module, cali_set, loss_type, iters):
    loss_func = Loss(module, p=2, loss_type=loss_type)
    optimizer = torch.optim.Adam(module.parameters())
    model.eval()  # Set the model to evaluation mode
    for i in range(iters):
        for data, _ in cali_set:
            model.zero_grad()
            output = model(data)
            quant_output = module(output)
            target_output = data  # Assuming the target output is the input itself for simplicity
            loss = loss_func(quant_output, target_output)
            loss.backward()
            optimizer.step()

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