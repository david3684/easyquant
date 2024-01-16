import quantizer, utils, torch
import torch.nn.functional as F
from quant import QModule, QModel
def reconstruct(model, calibration_set, reconstruction_method='layer', loss_type = 'mse' ):
    """
    Reconstruct the quantized model using the calibration set.

    :param model: The quantized model.
    :param calibration_set: The calibration set for reconstruction.
    :param reconstruction_method: The method of reconstruction ('layer_wise', 'block_wise', etc.)
    """
    for name, module in model.named_modules():
        if isinstance(module, QModule):
            print(f"Reconstructing layer: {name}")

            # 재구성 방법에 따라 적절한 재구성 함수 호출
            if reconstruction_method == 'layer':
                layer_reconstruction(model, module, calibration_set)
            # 다른 재구성 방법들에 대한 처리는 여기에 추가
            
            else:
                raise NotImplementedError(f"Reconstruction method '{reconstruction_method}' not implemented")

    print("Reconstruction completed.")
    
    
def layer_reconstruction(model, module, cali_set):
    
    return

def block_reconstruction():
    return


class Loss:
    def __init__(self, layer, p):
        self.layer = layer
    def __call__(self, pred, target):
        if type == 'mse':
            rec_loss = utils.lp_loss(pred, target, p=self.p) # p=2
        elif type == 'cosine':
            similarity = F.cosine_similarity(pred.view(-1), target.view(-1), dim=1)
            rec_loss = 1 - similarity.mean()