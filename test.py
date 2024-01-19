import easyquant.quant
import easyquant.quantizer
import torch, os
from tqdm import tqdm
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.hub as hub

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
    model = torch.hub.load('yhhhli/BRECQ', model='resnet18', pretrained=True)
    #model = models.resnet18(pretrained=True)
    num_classes=200
    #model.fc = nn.Linear(model.fc.in_features, num_classes)
    #nn.init.kaiming_uniform_(model.fc.weight, mode='fan_in', nonlinearity='relu')
    #model.fc.bias.data.fill_(0)
    qmodel = easyquant.quant.QModel(model)
    #for module in qmodel.modules():
    #    if hasattr(module, 'weight'):
    #        print(f'Layer: {module.__class__.__name__}, Weight data type: {module.weight.dtype}')
    #        # 스케일과 제로 포인트도 확인할 수 있음
    #        if isinstance(module, easyquant.quant.QModule):
    #            print(module.get_scale_zero_point())

    transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    #train_data = datasets.ImageFolder(root='./datasets/tiny-imagenet-200/train', transform=transforms)
    #train_loader = DataLoader(train_data, batch_size=128, shuffle=True)
    val_data = datasets.ImageFolder(root='./datasets/imagenet2012/val', transform=transforms)
    val_loader = DataLoader(val_data, batch_size=128, shuffle=True)
    #train_data = datasets.CIFAR100(root='./datasets', train=True, download=True, transform=transforms)
    #val_data = datasets.CIFAR100(root='./datasets', train=False, download=True, transform=transforms)
    #train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    #val_loader = DataLoader(val_data, batch_size=64, shuffle=True)
    
    if torch.cuda.is_available():
        qmodel.cuda()

    learning_rate = 0.001
    optimizer = torch.optim.Adam(qmodel.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    criterion = nn.CrossEntropyLoss()
    max_epoch = 50
    
    checkpoint_path = './checkpoints/checkpoint4.pth.tar'
    if os.path.isfile(checkpoint_path):
        print("=> Loading checkpoint")
        checkpoint = torch.load(checkpoint_path)
        start_epoch = load_checkpoint(checkpoint, qmodel, optimizer)
        print(f"=> Loaded checkpoint from epoch {start_epoch}")
    tloss, tacc = process_epoch(qmodel, criterion, val_loader, optimizer, trainmode = False)
    print('Train loss {:.3f} Train accuracy {:.1f}%'.format(tloss,tacc*100))
    #for epoch in range(max_epoch):
    #    tloss, tacc = process_epoch(qmodel, criterion, train_loader, optimizer, trainmode = True)
    #    print('Epoch {:d} completed. Train loss {:.3f} Train accuracy {:.1f}%'.format(epoch,tloss,tacc*100))
    #    if (epoch + 1) % 5 == 0:
    #        vloss, vacc = process_epoch(qmodel, criterion, val_loader, optimizer, trainmode = False)
    #        print('Epoch {:d} completed. Train loss {:.3f} Val loss {:.3f} Train accuracy {:.1f}% Test accuracy {:.1f}%'.format(epoch,tloss,vloss,tacc*100,vacc*100))
    #        save_checkpoint({
    #            'epoch': epoch + 1,
    #            'state_dict': model.state_dict(),
    #            'optimizer': optimizer.state_dict(),
    #        }, filename=checkpoint_path)
    