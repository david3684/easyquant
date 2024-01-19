import torch
import torch.nn as nn
import copy
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader
import os
from tqdm import tqdm
import warnings
import argparse
import quant, quantizer




def save_checkpoint(state, filename="checkpoint.pth.tar"):
    torch.save(state, filename)


def load_checkpoint(checkpoint, model, optimizer):
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    return checkpoint['epoch']

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
                    loss = criterion(outputs, labels)

            _, predicted = torch.max(outputs.data, 1)
            closs += loss.item() * images.size(0)  # 누적 손실
            total += labels.size(0)
            correct += (predicted == labels).sum().item()  # 누적 정확도

            # 각 배치 처리 후 tqdm 진행 바 업데이트
            tepoch.set_postfix(loss=(closs / total), acc_pct=(correct / total * 100))

    return (closs / total), (correct / total)

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Easy Quantization Package')

    # weight quantization bitwidth
    parser.add_argument('--w_n_bits', type=str, default='8',
                        choices=['uniform', 'adaround', 'custom'],
                        help='Method used for rounding during quantization.')

    # 재구성 방법
    parser.add_argument('--a_n_bits', type=str, default='8',
                        choices=['layer_wise', 'block_wise', 'custom'],
                        help='Reconstruction method to be used after quantization.')
    
    
    parser.add_argument('--init_method', type=int, default='min',
                        choices=['min', 'mse'],
                        help='Number of samples to use for calibration.')

    # calibration set size
    parser.add_argument('--cali_size', type=int, default=1024,
                        help='Number of samples to use for calibration.')
    
    # quantizer 
    parser.add_argument('--adaround', type=bool, default='true',
                        help='Apply adaptive rounding in reconstruction')
    
    # weight optimiztion method
    parser.add_argument('--w_optmod', type=int, default='mse',
                        help='Method used for optimizing quantized weights.')
    
    # activation optimiztion method
    parser.add_argument('--a_optmod', type=str, default='mse',
                        choices=['min_max', 'mse'],
                        help='Method used for optimizing quantized activations.')

    args = parser.parse_args()
    
    q_params = {'w_n_bits': args.w_n_bits, 'channel_wise': args.channel_wise, 'scale_method': args.init_wmode}
    
    model = torch.hub.load('yhhhli/BRECQ', model='resnet18', pretrained=True)

    num_classes=200

    qmodel = quant.QModel(model)

    transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_data = datasets.ImageFolder(root='./datasets/imagenet2012/val', transform=transforms)
    val_loader = DataLoader(val_data, batch_size=128, shuffle=True)

    
    if torch.cuda.is_available():
        qmodel.cuda()

    learning_rate = 0.001
    optimizer = torch.optim.Adam(qmodel.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    criterion = nn.CrossEntropyLoss()
    max_epoch = 50
    

    vloss, vacc = process_epoch(qmodel, criterion, val_loader, optimizer, trainmode = False)
    print('Val loss {:.3f} Val accuracy {:.1f}%'.format(vloss,vacc*100))
    