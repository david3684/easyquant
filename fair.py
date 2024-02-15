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

class UTKFaceDataset(Dataset):
    def __init__(self, csv_file, root_dir, subset='train', target = 'race', transform=None, train_pct=0.8, val_pct=0.1):
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



def evaluate(model, dataloader, task = 'race'):
    model.eval()
    correct = defaultdict(int)  
    total = defaultdict(int)  
    accuracy = {}  
    
    progress_bar = tqdm(dataloader, desc="Evaluating")
    
    for images, targets in progress_bar:
        # images = images.to('cuda')
        # targets = targets.to('cuda')
        # genders = genders.to('cuda')
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        for target, pred in zip(targets, predicted):
            total[target.item()] += 1
            correct[target.item()] += (pred == target).item()

        temp_overall_accuracy = sum(correct.values()) / sum(total.values()) * 100
        progress_bar.set_description(f"Evaluating (Accuracy: {temp_overall_accuracy:.2f}%)")

    accuracies = []
    if task == 'race':    
        for race in total:
            race_accuracy = 100 * correct[race] / total[race]
            accuracy[race] = race_accuracy
            accuracies.append(race_accuracy)
            print(f'Accuracy for race {race}: {race_accuracy:.2f}%')
    else:
        for gender in total:
            gender_accuracy = 100 * correct[gender] / total[gender]
            accuracy[gender] = gender_accuracy
            accuracies.append(gender_accuracy)
            str_gender = 'Male' if gender==0 else 'Female'
            print(f'Accuracy for gender {str_gender}: {gender_accuracy:.2f}%')
    overall_accuracy = sum(correct.values()) / sum(total.values()) * 100
    accuracy_std = np.std(list(accuracy.values()))
    print(f'Overall Accuracy: {overall_accuracy:.2f}%')
    print(f'Accuracy Standard Deviation: {accuracy_std:.2f}%')

    return accuracy, overall_accuracy, accuracy_std

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Parameters for pruning experiements')
    parser.add_argument('--target',type=str, default='race', help='evaluation target')
    parser.add_argument('--dataset',type=str, default='FairFace', help='Dataset')
    args = parser.parse_args()

    transform = transforms.Compose(([ 
                                    transforms.Resize((224, 224)),
                                                transforms.ToTensor(), 
                                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                                    ]))
    if args.dataset == 'FairFace':    
        dataset = FairFaceDataset(csv_file='/workspace/fair/FairGRAPE/csv/FairFace.csv', root_dir='/workspace/datasets/fairface', transform=transform, target=args.target)
        dataloader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4)
    elif args.dataset == 'UTKFace':
        dataset = UTKFaceDataset(csv_file='/workspace/fair/FairGRAPE/csv/UTKFace_labels.csv', root_dir='/workspace/datasets/utkcropped', target = args.target, subset='test', transform=transform)
        dataloader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4)
    
    # model = models.resnet34(pretrained=False)
    model = models.MobileNetV2(num_classes=4)
    if args.target == 'race':
        # model.fc = torch.nn.Linear(model.fc.in_features, 7) 
        model.load_state_dict(torch.load('/workspace/trained/trained_model/Full/UTKFace_Full_race_bygender_mobilenetv2_0_10000.pt'))
    else:
        model.fc = torch.nn.Linear(model.fc.in_features, 2) 
        model.load_state_dict(torch.load('/workspace/trained/trained_model/Full/FairFace_Full_gender_byrace_resnet34_0_60.pt'))
    # model.to('cuda')
    model.eval() 
    group_accuracy, overall_accuracy, std = evaluate(model, dataloader, 'race')
    model_copy = copy.deepcopy(model)
    qmodel = quant.QModel(model_copy, w_n_bits=8, init_method='mse')
    # qmodel.to('cuda')
    # For calibration
    # calibration_dataset = FairFaceDataset(csv_file='/workspace/fair/FairGRAPE/csv/FairFace.csv', root_dir='/workspace/datasets/fairface', transform=transform,target=args.target, sample=1000)
    # calibration_dataloader = DataLoader(calibration_dataset, batch_size=64, shuffle=True, num_workers=4)
    # reconstruct(qmodel, model, calibration_dataloader, adaround=True)
    qmodel.eval()
    quantized_group_accuracy, quantized_overall_accuracy, q_std = evaluate(qmodel, dataloader, 'race')
    group_accuracy, overall_accuracy, std = evaluate(model, dataloader, 'gender')
    quantized_group_accuracy, quantized_overall_accuracy, q_std = evaluate(qmodel, dataloader, 'gender')
    

