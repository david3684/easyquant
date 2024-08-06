import torch
import timm
import torch.nn as nn
import copy
from torchvision import models, datasets, transforms
import torch.optim as optim
from torch.utils.data import DataLoader
import os
from tqdm import tqdm
import warnings
import argparse
import json
import quant
import quantizer
import calibration
from recon import reconstruct
import torch.onnx


def train_model(model, train_loader, criterion, optimizer, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, labels in tqdm(train_loader):
            images = images.to('cuda')
            labels = labels.to('cuda')

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(
            f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")


def evaluate_model(model, data_loader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(data_loader):
            images = images.to('cuda')
            labels = labels.to('cuda')

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    return accuracy


model_name = 'vit_base_patch16_224'
model = timm.create_model(model_name, pretrained=True)

num_classes = 10
model.head = nn.Linear(model.head.in_features, num_classes)

transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dataset = datasets.CIFAR10(
    root='data', train=True, transform=transforms, download=True)
train_loader = DataLoader(train_dataset, batch_size=32,
                          shuffle=True, num_workers=4)

val_dataset = datasets.CIFAR10(
    root='data', train=False, transform=transforms, download=True)
val_loader = DataLoader(val_dataset, batch_size=32,
                        shuffle=False, num_workers=4)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)


model.to('cuda')

train_model(model, train_loader, criterion, optimizer, num_epochs=10)

accuracy = evaluate_model(model, val_loader)
print(
    f'Accuracy of the model on the CIFAR-10 validation images: {accuracy * 100:.2f}%')

model.eval()

input_tensor = torch.randn(1, 3, 224, 224)

model_copy = copy.deepcopy(model)

qmodel = quant.QModel(
    model_copy, w_n_bits=8,
    init_method='minmax')
