import torch
import torchvision.models as models
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from PIL import Image
import os
from collections import defaultdict
from tqdm import tqdm
import quant
from recon import reconstruct
import calibration
import copy
import numpy as np
import argparse

def load_checkpoint(checkpoint_dir, model, optimizer):
    # 체크포인트 경로에서 가장 최근의 파일을 찾습니다.
    checkpoints = [checkpoint for checkpoint in os.listdir(checkpoint_dir) if checkpoint.startswith("celeba_checkpoint_epoch_")]
    if checkpoints:
        latest_checkpoint = max(checkpoints, key=lambda x: int(x.split('_')[-1].split('.')[0]))
        checkpoint_path = os.path.join(checkpoint_dir, latest_checkpoint)
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resuming training from epoch {start_epoch}")
    else:
        start_epoch = 0
        print("No checkpoint found. Starting from scratch.")
    return start_epoch

def create_dataset(dataset, subset, target, transform, train_pct=0.8, val_pct=0.1, sample=None):
    if dataset == 'FairFace':
        return FairFaceDataset(csv_file='/workspace/fair/FairGRAPE/csv/FairFace.csv', root_dir='/workspace/datasets/fairface', transform=transform, target=target, sample=sample)
    elif dataset == 'UTKFace':
        return UTKFaceDataset(csv_file='/workspace/fair/FairGRAPE/csv/UTKFace_labels.csv', root_dir='/workspace/datasets/utkcropped', target = target, subset=subset, transform=transform, train_pct=train_pct, val_pct=val_pct, sample=sample)

class UTKFaceDataset(Dataset):
    def __init__(self, csv_file, root_dir, subset='train', target = 'race', transform=None, train_pct=0.8, val_pct=0.1, sample=None):
        # Read and process the CSV file
        df = pd.read_csv(csv_file)
        # Extract filenames and adjust paths
        df['face_name_align'] = df['face_name_align'].apply(lambda x: os.path.basename(x))
        # Shuffle and split the dataset
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        total_size = len(df)
        train_size = int(total_size * train_pct)
        val_size = int(total_size * val_pct)
        
        if subset == 'train':
            self.annotations = df.iloc[:train_size].reset_index(drop=True)
        elif subset == 'val':
            self.annotations = df.iloc[train_size:train_size + val_size].reset_index(drop=True)
        else:  # Assuming 'test'
            self.annotations = df.iloc[train_size + val_size:].reset_index(drop=True)
        if sample is not None:
            self.annotations = self.annotations.sample(n=sample).reset_index(drop=True)
        self.root_dir = root_dir
        self.transform = transform
        self.target = target
    
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, index):
        row = self.annotations.iloc[index]
        # Construct the image path using the root_dir and the filename
        img_path = os.path.join(self.root_dir, row['face_name_align'])
        image = Image.open(img_path).convert('RGB')
        # Assuming gender is coded as 'Male'/'Female', convert to 0/1
        race_label = self.annotations.iloc[index, 3]
        gender_label = 0 if row['gender'] == 'Male' else 1
        race = self.race_to_label(race_label)
        
        if self.transform:
            image = self.transform(image)
        
        if self.target == 'gender':
            return image, gender_label
        else: return image, race
    
    @staticmethod
    def race_to_label(race_label):
        race_dict = {
            'White': 0,
            'Black': 1,
            'East Asian': 2,
            'Indian': 3,
        }
        return race_dict.get(race_label, -1)


class CelebADataset(Dataset):
    def __init__(self, csv_file, root_dir, subset='train', transform=None, train_pct=0.8, val_pct=0.1):
        # Read the CSV file
        self.df = pd.read_csv(csv_file)
        
        # Split the dataset
        self.df = self.df.sample(frac=1).reset_index(drop=True)  # Shuffle the dataset
        total_size = len(self.df)
        train_size = int(total_size * train_pct)
        val_size = int(total_size * val_pct)
        test_size = total_size - train_size - val_size
        
        if subset == 'train':
            self.annotations = self.df[:train_size]
        elif subset == 'val':
            self.annotations = self.df[train_size:train_size + val_size]
        else:  # 'test'
            self.annotations = self.df[train_size + val_size:]
        
        self.root_dir = root_dir
        self.transform = transform
    
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, index):
        row = self.annotations.iloc[index]
        img_path = os.path.join(self.root_dir, row['face_name_align'])
        image = Image.open(img_path)
        
        gender_label = row.iloc[-1]  # Assuming 'gender' is the label column
        
        if self.transform:
            image = self.transform(image)
        
        return image, gender_label


class FairFaceDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None, target = 'gender', sample=None):
        self.annotations = pd.read_csv(csv_file)
        self.annotations = self.annotations[self.annotations['file'].str.startswith('val_')]
        self.root_dir = root_dir
        self.transform = transform
        self.target = target
        if sample is not None:
            self.annotations = self.annotations.sample(n=sample).reset_index(drop=True)

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        file_name = self.annotations.iloc[index, 0].split('_')[1]
        img_path = os.path.join(self.root_dir, "val", file_name)
        image = Image.open(img_path)
        
        race_label = self.annotations.iloc[index, 3]
        gender_label = self.annotations.iloc[index, 2]
        race = self.race_to_label(race_label)
        gender = 0 if gender_label == 'Male' else 1
        
        if self.transform:
            image = self.transform(image)

        if self.target == 'gender':
            return image, gender
        else: return image, race

    @staticmethod
    def race_to_label(race_label):
        race_dict = {
            'White': 0,
            'Black': 1,
            'Latino_Hispanic': 2,
            'East Asian': 3,
            'Southeast Asian': 4,
            'Indian': 5,
            'Middle Eastern': 6
        }
        return race_dict.get(race_label, -1)


def start_train(model, train_loader, val_loader, num_epochs, loss, optimizer, device):
    checkpoint_dir = './checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)  # 체크포인트 디렉토리 생성
    start_epoch = load_checkpoint(checkpoint_dir, model, optimizer)
    
    for epoch in range(start_epoch, num_epochs):
        evaluate(model, train_loader, mode='train', loss_fn=loss, optimizer=optimizer, device = device)
        with torch.no_grad():
            overall_accuracy = evaluate(model, val_loader, mode='eval', task='race', device = device)
        
        print(f"Validation - Overall Accuracy: {overall_accuracy:.2f}%\n")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, os.path.join(checkpoint_dir, f'celeba_checkpoint_epoch_{epoch}.pt'))
        print(f'Checkpoint saved for epoch {epoch}')  

        
    
from tqdm import tqdm
from collections import defaultdict
import numpy as np
import torch

def evaluate(model, dataloader, mode='train', loss_fn=None, optimizer=None, device='cuda', task='race'):
    if mode == 'train':
        model.train()
    else:
        model.eval()
    
    model.to(device)
    correct = defaultdict(int)
    total = defaultdict(int)
    running_loss = 0.0
    progress_bar = tqdm(dataloader, desc="Processing")

    for images, targets in progress_bar:
        images, targets = images.to(device), targets.to(device)
        
        if mode == 'train':
            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        else:
            with torch.no_grad():
                outputs = model(images)
        
        _, predicted = torch.max(outputs.data, 1)
        for label, prediction in zip(targets, predicted):
            if label == prediction:
                correct[label.item()] += 1
            total[label.item()] += 1
        
        overall_correct = sum(correct.values())
        overall_total = sum(total.values())
        overall_accuracy = 100. * overall_correct / overall_total
        
        progress_bar.set_description(f"{'Training' if mode == 'train' else 'Evaluating'}")
        progress_bar.set_postfix(loss=(running_loss / (overall_total / dataloader.batch_size)) if mode == 'train' else None, accuracy=overall_accuracy)

    if mode != 'train':
        for cls in total.keys():
            cls_accuracy = 100. * correct[cls] / total[cls]
            print(f'Accuracy for class {cls}: {cls_accuracy:.2f}%')

    if mode == 'train':
        loss_text = f"{running_loss / (overall_total / dataloader.batch_size):.4f}"
    else:
        loss_text = "-"

    print(f'\n{"Training" if mode == "train" else "Evaluation"} completed: Loss: {loss_text}, Accuracy: {overall_accuracy:.2f}%')


    return overall_accuracy


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Parameters for pruning experiements')
    parser.add_argument('--target',type=str, default='race', help='evaluation target')
    parser.add_argument('--dataset',type=str, default='FairFace', help='Dataset')
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose(([ 
                                    transforms.Resize((224, 224)),
                                                transforms.ToTensor(), 
                                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                                    ]))
    if args.dataset == 'FairFace':    
        dataset = create_dataset(dataset=args.dataset, target='race', transform=transform, sample=None)
        dataloader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4)
    elif args.dataset == 'UTKFace':
        train_dataset = create_dataset(dataset=args.dataset, target='race', transform=transform, sample=None, subset='train')
        train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
        val_dataset = create_dataset(dataset=args.dataset, target='race', transform=transform, sample=None, subset='val')
        val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4)
        test_dataset = create_dataset(dataset=args.dataset, target='race', transform=transform, sample=None, subset='test')
        test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4)
    
    model = models.resnet18(pretrained = True)
    model.to(device)
    loss = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)
    if args.target == 'race':
        model.fc = torch.nn.Linear(model.fc.in_features, 4) 
        start_train(model, train_dataloader, val_dataloader, num_epochs=11, loss=loss, optimizer=optimizer, device=device )
    else:
        model.fc = torch.nn.Linear(model.fc.in_features, 2) 
        model.load_state_dict(torch.load('/workspace/trained/trained_model/Full/FairFace_Full_gender_byrace_resnet34_0_60.pt'))

    model.eval() 
    overall_accuracy = evaluate(model, test_dataloader, mode='test', task= 'race', device=device)
    model_copy = copy.deepcopy(model)
    qmodel = quant.QModel(model_copy, w_n_bits=8, init_method='mse')
    
    qmodel.to(device)
    
    calibration_dataset = create_dataset(dataset=args.dataset, target='race', transform=transform, sample=1000, subset='test')
    calibration_dataloader = DataLoader(calibration_dataset, batch_size=64, shuffle=False, num_workers=4)
    reconstruct(qmodel, model, calibration_dataloader, adaround=True)
    torch.save(qmodel.state_dict(),f'./checkpoints/qunatized/quantized_{args.dataset}_{args.target}.pt')
    qmodel.eval()
    quantized_overall_accuracy= evaluate(qmodel, test_dataloader, 'race')
    group_accuracy, overall_accuracy, std = evaluate(model, dataloader, 'gender')
    quantized_overall_accuracy = evaluate(qmodel, dataloader, 'gender')
    

