import torch
import torch.nn as nn
import copy
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader
import os
from tqdm import tqdm
import warnings
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Easy Quantization Package')

    # 모델 경로
    parser.add_argument('--model_path', type=str, help='Path to the saved model file')

    # TFRecord 디렉토리 경로
    parser.add_argument('--tfrecord_dir', type=str, required=True,
                        help="Path to a directory containing ImageNet TFRecords.\n"
                             "This folder should contain files starting with:\n"
                             "'train*': for training records and 'validation*': for validation records")

    # CUDA 사용 여부
    parser.add_argument('--use_cuda', action='store_true', default=False,
                        help='Add this flag to run the test on GPU.')

    # 양자화 중 라운딩 방법
    parser.add_argument('--round_method', type=str, default='uniform',
                        choices=['uniform', 'adaround', 'custom'],
                        help='Method used for rounding during quantization.')

    # 재구성 방법
    parser.add_argument('--reconstruction', type=str, default='layer_wise',
                        choices=['layer_wise', 'block_wise', 'custom'],
                        help='Reconstruction method to be used after quantization.')

    # 가중치 양자화 여부
    parser.add_argument('--weight_quant', action='store_true', default=False,
                        help='Enable weight quantization.')

    # 활성화 양자화 여부
    parser.add_argument('--act_quant', action='store_true', default=False,
                        help='Enable activation quantization.')

    # 양자화 비트 수
    parser.add_argument('--n_bits', type=int, default=8,
                        help='Number of bits to use for quantization.')

    # 캘리브레이션 데이터셋 크기
    parser.add_argument('--calibration_size', type=int, default=1024,
                        help='Number of samples to use for calibration.')

    # 양자화 스케일 방법
    parser.add_argument('--scale_method', type=str, default='mse',
                        choices=['min_max', 'mse', 'cosine'],
                        help='Method used for scaling during quantization.')

    args = parser.parse_args()


########################################    
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