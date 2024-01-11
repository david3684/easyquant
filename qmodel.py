import torch
import torch.nn as nn
import copy
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader
import os
from tqdm import tqdm
import warnings

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  # Arrange GPU devices starting from 0
os.environ["CUDA_VISIBLE_DEVICES"]= "0"  

def lp_loss(pred, tgt, p=2.0, reduction='none'):
    """
    loss function measured in L_p Norm
    """
    if reduction == 'none':
        return (pred-tgt).abs().pow(p).sum(1).mean()
    else:
        return (pred-tgt).abs().pow(p).mean()
    
class QModel:
    def __init__(self, model, symmetric, delta, zero_point, scale_method):
        self.model = model
        self.sym = symmetric
        self.delta = None
        self.zero_point = None
        

    def quantize_layer(self, layer):
        weight = layer.weight.data
        print("Original Weights: {}".format(weight))

        
        delta, zero_point = self.init_quantization_scale(weight, channel_wise)

        
        quantized_weight = torch.round((weight - zero_point) / delta).int()
        print("Quantized Weights: {}".format(quantized_weight))

        
        dequantized_weight = (quantized_weight.float() * delta) + zero_point
        print("DeQuantized Weights: {}".format(dequantized_weight))

        quantized_layer = copy.deepcopy(layer)
        quantized_layer.weight.data = dequantized_weight

        return quantized_layer

    def _apply_quantization(self, module):
        for child_name, child_module in module.named_children():
            if isinstance(child_module, (nn.Conv2d, nn.Linear)):  
                quantized_layer = self.quantize_layer(child_module)
                setattr(module, child_name, quantized_layer)
            else:
                self._apply_quantization(child_module)  
                
    def init_quantization_scale(self, x: torch.Tensor, channel_wise: bool = False):
        delta, zero_point = None, None
        if channel_wise:
            x_clone = x.clone().detach()
            n_channels = x_clone.shape[0]
            if len(x.shape) == 4:
                x_max = x_clone.abs().max(dim=-1)[0].max(dim=-1)[0].max(dim=-1)[0]
            else:
                x_max = x_clone.abs().max(dim=-1)[0]
            delta = x_max.clone()
            zero_point = x_max.clone()
            # determine the scale and zero point channel-by-channel
            for c in range(n_channels):
                delta[c], zero_point[c] = self.init_quantization_scale(x_clone[c], channel_wise=False)
            if len(x.shape) == 4:
                delta = delta.view(-1, 1, 1, 1)
                zero_point = zero_point.view(-1, 1, 1, 1)
            else:
                delta = delta.view(-1, 1)
                zero_point = zero_point.view(-1, 1)
        else:
            if 'max' in self.scale_method:
                x_min = min(x.min().item(), 0)
                x_max = max(x.max().item(), 0)
                if 'scale' in self.scale_method:
                    x_min = x_min * (self.n_bits + 2) / 8
                    x_max = x_max * (self.n_bits + 2) / 8

                x_absmax = max(abs(x_min), x_max)
                if self.sym:
                    x_min, x_max = -x_absmax if x_min < 0 else 0, x_absmax

                delta = float(x_max - x_min) / (self.n_levels - 1)
                if delta < 1e-8:
                    warnings.warn('Quantization range close to zero: [{}, {}]'.format(x_min, x_max))
                    delta = 1e-8

                zero_point = round(-x_min / delta)
                delta = torch.tensor(delta).type_as(x)

            elif self.scale_method == 'mse':
                x_max = x.max()
                x_min = x.min()
                best_score = 1e+10
                for i in range(80):
                    new_max = x_max * (1.0 - (i * 0.01))
                    new_min = x_min * (1.0 - (i * 0.01))
                    x_q = self.quantize(x, new_max, new_min)

                    score = lp_loss(x, x_q, p=2.4, reduction='all')
                    if score < best_score:
                        best_score = score
                        delta = (new_max - new_min) / (2 ** self.n_bits - 1)
                        zero_point = (- new_min / delta).round()
            else:
                raise NotImplementedError
            
    def quantize(self):
        self._apply_quantization(self.model)
        return self.model

    
    

###############################################    
###############################################    
###############################################    
model = models.resnet18(pretrained=True)
qmodel = QModel(model)
quantized_model = qmodel.quantize()
#model = quantized_model

num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, 10)

model = model.to("cuda")

data_dir = "./datasets"
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

save_dir = "./checkpoints"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
    
transforms = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])
])

train_data = datasets.CIFAR10(root=data_dir, download=True, train=True, transform=transforms)
train_loader =DataLoader(train_data, batch_size=128, shuffle=True)
val_data = datasets.CIFAR10(root=data_dir, download=True, train=False, transform=transforms)
val_loader =DataLoader(val_data, batch_size=128, shuffle=True)
dataiter = iter(val_loader)
images, labels = next(dataiter)

print('Dataset Load Completed')

def process_epoch(model, criterion, loader, optimizer=None, trainmode=True):
    if trainmode:
        model.train()
    else:
        model.eval()
    
    closs = 0
    correct = 0
    total = 0
    with tqdm(loader, unit='batch') as tepoch:
        for images, labels in tepoch:
            if torch.cuda.is_available():
                #print("Moving to CUDA")
                images = images.cuda()
                labels = labels.cuda()
            if trainmode:
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
            else:
                with torch.no_grad():
                    outputs = model(images)
                    loss = criterion(outputs, labels) # average loss from all the samples
        _, predicted = torch.max(outputs.data, 1) # value, index of second dimension(prob. of classes)
        
        closs   += loss.item() * labels.size(0) # 배치 샘플 수 x 배치의 loss
        total   += labels.size(0) #배치 크기
        correct += (predicted == labels).sum().item() # 배치 길이 * boolean 중에 맞는 거 개수 
        tepoch.set_postfix(loss=closs/total, acc_pct=correct/total*100)

    return (closs/total), (correct/total)
                
num_classes = 10
max_epoch = 50
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

model.train()
for epoch in range(max_epoch):
    tloss, tacc = process_epoch(model, criterion, train_loader, optimizer, trainmode=True)
    vloss, vacc = process_epoch(model, criterion, val_loader, optimizer, trainmode=False)
    print('Epoch {:d} completed. Train loss {:.3f} Val loss {:.3f} Train accuracy {:.1f}% Test accuracy {:.1f}%'.format(epoch,tloss,vloss,tacc*100,vacc*100))
    scheduler.step()
    if(epoch+1)%5 == 0:
        save_path = os.path.join(save_dir,f'resnet50_epoch_{epoch+1}.pth')
        torch.save(model.state_dict(), save_path)
        print(f'Model saved at {save_path}')


model.eval()
val_loss, val_accuracy = process_epoch(model, criterion, val_loader, trainmode=False)

print(f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy * 100:.2f}%')

##########################
model = models.resnet18(pretrained=False)
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, 10)  # 예시에서는 10개의 클래스로 설정

# 2. 체크포인트 로드
# 저장된 체크포인트 파일 경로
checkpoint_path = os.path.join(save_dir, 'resnet50_epoch_50.pth')  # 예시 경로

# 체크포인트 파일이 존재하는 경우에만 로드
if os.path.isfile(checkpoint_path):
    model.load_state_dict(torch.load(checkpoint_path))
    print(f'Checkpoint loaded: {checkpoint_path}')
else:
    print(f'No checkpoint found at: {checkpoint_path}')
    
model.eval()