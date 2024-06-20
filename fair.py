import torch
from torchvision.models import resnet18, ResNet18_Weights, resnet34, ResNet34_Weights
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
import logging
import datetime

current_time = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
log_filename = f'model_training_{current_time}.log'
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 파일 핸들러 추가 (로그를 파일에 저장)
file_handler = logging.FileHandler(log_filename)
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(file_handler)

def load_checkpoint(checkpoint_dir, model, optimizer):
    checkpoints = [checkpoint for checkpoint in os.listdir(checkpoint_dir) if checkpoint.startswith("FairFace_")]
    if checkpoints:
        latest_checkpoint = max(checkpoints, key=lambda x: int(x.split('_')[-1].split('.')[0]))
        checkpoint_path = os.path.join(checkpoint_dir, latest_checkpoint)
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        logger.info(f"Resuming training from epoch {start_epoch}")
    else:
        start_epoch = 0
        logger.info("No checkpoint found. Starting from scratch.")
    return start_epoch

def create_dataset(dataset, subset, target, transform, train_pct=0.8, val_pct=0.1, sample=None, random_sample=True):
    if dataset == 'FairFace':
        return FairFaceDataset(csv_file='/workspace/fair/FairGRAPE/csv/FairFace.csv', root_dir='/workspace/datasets/fairface', transform=transform, target=target, sample=sample, random_sample=random_sample, subset=subset)
    elif dataset == 'UTKFace':
        return UTKFaceDataset(csv_file='/workspace/fair/FairGRAPE/csv/UTKFace_labels.csv', root_dir='/workspace/datasets/utkcropped', target = target, subset=subset, transform=transform, train_pct=train_pct, val_pct=val_pct, sample=sample)

class UTKFaceDataset(Dataset):
    def __init__(self, csv_file, root_dir, subset='train', target = 'race', transform=None, train_pct=0.8, val_pct=0.1, sample=None, sample_random=True):
        df = pd.read_csv(csv_file)
        df['face_name_align'] = df['face_name_align'].apply(lambda x: os.path.basename(x))
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        total_size = len(df)
        train_size = int(total_size * train_pct)
        val_size = int(total_size * val_pct)
        
        if subset == 'train':
            self.annotations = df.iloc[:train_size].reset_index(drop=True)
        elif subset == 'val':
            self.annotations = df.iloc[train_size:train_size + val_size].reset_index(drop=True)
        else:
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
        img_path = os.path.join(self.root_dir, row['face_name_align'])
        image = Image.open(img_path).convert('RGB')
        race_label = self.annotations.iloc[index, 3]
        gender_label = 0 if row['gender'] == 'Male' else 1
        race = self.race_to_label(race_label)
        
        if self.transform:
            image = self.transform(image)
        
        return image, race, gender_label
    
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
        self.df = pd.read_csv(csv_file)
        self.df = self.df.sample(frac=1).reset_index(drop=True)
        total_size = len(self.df)
        train_size = int(total_size * train_pct)
        val_size = int(total_size * val_pct)
        test_size = total_size - train_size - val_size
        
        if subset == 'train':
            self.annotations = self.df[:train_size]
        elif subset == 'val':
            self.annotations = self.df[train_size:train_size + val_size]
        else:
            self.annotations = self.df[train_size + val_size:]
        
        self.root_dir = root_dir
        self.transform = transform
    
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, index):
        row = self.annotations.iloc[index]
        img_path = os.path.join(self.root_dir, row['face_name_align'])
        image = Image.open(img_path)
        
        gender_label = row.iloc[-1]
        
        if self.transform:
            image = self.transform(image)
        
        return image, gender_label


class FairFaceDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None, target='gender', sample=None, random_sample=True, subset='train'):
        self.annotations = pd.read_csv(csv_file)
        if subset == 'train':
            self.annotations = self.annotations[self.annotations['file'].str.startswith('train_')]
        elif subset == 'val':
            self.annotations = self.annotations[self.annotations['file'].str.startswith('val_')]
        
        self.root_dir = root_dir
        self.transform = transform
        self.target = target
        self.subset = subset
        self.sample = sample
        if sample is not None:
            if random_sample:
                self.annotations = self.annotations.sample(n=sample).reset_index(drop=True)
            else:
                race_counts = self.annotations['race'].value_counts(normalize=True) * sample
                sampled_annotations = pd.DataFrame()
                for race, count in race_counts.items():
                    race_samples = self.annotations[self.annotations['race'] == race].sample(n=int(count))
                    sampled_annotations = pd.concat([sampled_annotations, race_samples])
                self.annotations = sampled_annotations.reset_index(drop=True)

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        file_name = self.annotations.iloc[index, 0]
        file_name = file_name.split('_')[1]
        img_path = os.path.join(self.root_dir, self.subset, file_name)
        
        image = Image.open(img_path)
        
        race_label = self.annotations.iloc[index, 3]
        gender_label = self.annotations.iloc[index, 2]
        race = self.race_to_label(race_label)
        gender = 0 if gender_label == 'Male' else 1
        
        if self.transform:
            image = self.transform(image)
        if self.sample is not None:
            return image, race if self.target == 'race' else gender
        return image, race, gender

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


def start_train(model, train_loader, val_loader, dataset, num_epochs, loss, optimizer, device, target = 'race'):
    checkpoint_dir = '../checkpoints'
    checkpoint_dir = os.path.join(checkpoint_dir, f'{dataset}_checkpoints_{target}')
    os.makedirs(checkpoint_dir, exist_ok=True)
    start_epoch = load_checkpoint(checkpoint_dir, model, optimizer)
    
    for epoch in range(start_epoch, num_epochs):
        evaluate(model, train_loader, mode='train', loss_fn=loss, evaluation_target=target, optimizer=optimizer, device = device)
        with torch.no_grad():
            overall_accuracy = evaluate(model, val_loader, mode='eval', evaluation_target=target, device = device)
        
        logger.info(f"Validation - Overall Accuracy: {overall_accuracy:.2f}%")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, os.path.join(checkpoint_dir, f'{dataset}_checkpoint_{target}_epoch_{epoch}.pt'))
        logger.info(f'Checkpoint saved for epoch {epoch}')  

from tqdm import tqdm
from collections import defaultdict
import numpy as np
import torch

def evaluate(model, dataloader, mode='train', loss_fn=None, optimizer=None, device='cuda', evaluation_target='gender'):
    model.to(device)
    if mode == 'train':
        model.train()
    else:
        model.eval()

    accuracy_details = defaultdict(int)
    total_details = defaultdict(int)
    running_loss = 0.0
    total_samples = 0
    correct_predictions = 0

    with tqdm(dataloader, desc=f"{mode.capitalize()} Mode") as tqdm_loader:
        for images, race_labels, gender_labels in tqdm_loader:
            images, race_labels, gender_labels = images.to(device), race_labels.to(device), gender_labels.to(device)

            if mode == 'train':
                optimizer.zero_grad()

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            target_labels = gender_labels if evaluation_target == 'gender' else race_labels

            if mode == 'train':
                loss = loss_fn(outputs, target_labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            total_samples += images.size(0)
            correct_predictions += (predicted == target_labels).sum().item()

            for i in range(len(target_labels)):
                key = f"race_{race_labels[i].item()}"
                accuracy_details[key] += (predicted[i] == target_labels[i]).item()
                total_details[key] += 1

                key = f"gender_{gender_labels[i].item()}"
                accuracy_details[key] += (predicted[i] == target_labels[i]).item()
                total_details[key] += 1

            current_loss = running_loss / total_samples
            current_accuracy = 100 * correct_predictions / total_samples
            tqdm_loader.set_postfix(loss=f"{current_loss:.4f}", accuracy=f"{current_accuracy:.2f}%")

    logger.info("Gender Accuracy:")
    for i in sorted([key for key in accuracy_details.keys() if "gender" in key], key=lambda x: int(x.split('_')[1])):
        total = total_details[i]
        accuracy = 100.0 * accuracy_details[i] / total
        logger.info(f"{i.replace('_', ' ')} Accuracy: {accuracy:.2f}%")

    logger.info("Race groups:")
    for i in sorted([key for key in accuracy_details.keys() if "race" in key], key=lambda x: int(x.split('_')[1])):
        total = total_details[i]
        accuracy = 100.0 * accuracy_details[i] / total
        logger.info(f"{i.replace('_', ' ')} Accuracy: {accuracy:.2f}%")

    race_accuracies = [100.0 * accuracy_details[key] / total_details[key] for key in accuracy_details if "race" in key]
    gender_accuracies = [100.0 * accuracy_details[key] / total_details[key] for key in accuracy_details if "gender" in key]
    race_accuracy_std = np.std(race_accuracies)
    gender_accuracy_std = np.std(gender_accuracies)

    logger.info(f"Accuracy std among race groups: {race_accuracy_std:.2f}%")
    logger.info(f"Accuracy std among gender groups: {gender_accuracy_std:.2f}%")

    overall_accuracy = 100 * correct_predictions / total_samples
    logger.info(f"{mode.capitalize()} 손실: {running_loss / len(dataloader):.4f}, 전체 정확도: {overall_accuracy:.2f}%")

    return overall_accuracy

def main():
    parser = argparse.ArgumentParser(description='Parameters for model training and evaluation')
    parser.add_argument('--target', type=str, default='race', help='Evaluation target (race or gender)')
    parser.add_argument('--dataset', type=str, default='FairFace', help='Dataset to use (FairFace or UTKFace)')
    parser.add_argument('--from_train', action='store_true')
    parser.add_argument('--model_path', type=str, default='../trained/trained_model/Full')
    args = parser.parse_args()

    torch.cuda.set_device(1)
    device_id = 1  # 사용할 GPU ID 설정 (여기서는 1번 GPU)
    torch.cuda.set_device(device_id)
    device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    if(args.from_train):
        train_dataset = create_dataset(dataset=args.dataset, target=args.target, transform=transform, subset='train')
        train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
        val_dataset = create_dataset(dataset=args.dataset, target=args.target, transform=transform, subset='val')
        val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4)
    
    test_dataset = create_dataset(dataset=args.dataset, target=args.target, transform=transform, subset='val')
    test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4)
    
    weights = ResNet34_Weights.IMAGENET1K_V1
    model = resnet34(weights=weights)
    if args.target == 'race':
        num_classes = 7
    else:
        num_classes = 2 
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes) 
    model.to(device)
    if not args.from_train:
        state_dict = torch.load(args.model_path)
        model.load_state_dict(state_dict)

    loss = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    if args.from_train:
        start_train(model, train_dataloader, val_dataloader, dataset = args.dataset, num_epochs=11, loss=loss, optimizer=optimizer, device=device)
    
    model.eval()
    overall_accuracy = evaluate(model, test_dataloader, mode='test', device=device, evaluation_target=args.target)
    logger.info(f"Test Overall Accuracy for {args.target} on {args.dataset}: {overall_accuracy:.2f}%")

    model_copy = copy.deepcopy(model)
    model_copy_2 = copy.deepcopy(model)
    qmodel = quant.QModel(model_copy, w_n_bits=4, init_method = 'mse')
    qmodel_copy = quant.QModel(model_copy_2, w_n_bits=4, init_method = 'mse')
    
    overall_accuracy = evaluate(qmodel, test_dataloader, mode='test', device=device, evaluation_target=args.target)
    logger.info(f"Quantized Overall Accuracy for {args.target} on {args.dataset}: {overall_accuracy:.2f}%")
    
    cali_set_random = create_dataset(dataset=args.dataset, target=args.target, transform=transform, subset='val', sample=1024, random_sample=True)
    cali_set_uniform = create_dataset(dataset=args.dataset, target=args.target, transform=transform, subset='val', sample=1024, random_sample=False)
    random_cali_loader = DataLoader(cali_set_random, batch_size=32, shuffle=False, num_workers=4)
    uniform_cali_loader = DataLoader(cali_set_uniform, batch_size=32, shuffle=False, num_workers=4)
    
    reconstruct(qmodel=qmodel, fpmodel=model, calibration_set=random_cali_loader, adaround=True, recon_act=True)
    overall_accuracy = evaluate(qmodel, test_dataloader, mode='test', device=device, evaluation_target=args.target)
    logger.info(f"Quantized Overall Accuracy after random reconstruction for {args.target} on {args.dataset}: {overall_accuracy:.2f}%")

    reconstruct(qmodel=qmodel_copy, fpmodel=model, calibration_set=uniform_cali_loader, adaround=True, recon_act=True)
    overall_accuracy = evaluate(qmodel_copy, test_dataloader, mode='test', device=device, evaluation_target=args.target)
    logger.info(f"Quantized Overall Accuracy after uniform reconstruction for {args.target} on {args.dataset}: {overall_accuracy:.2f}%")

if __name__ == '__main__':
    main()
