import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np

def sample_calibration_set(dataset_path, calibration_size, transform=None, batch_size=32):
    """
    Sample a calibration set from the given dataset.

    :param dataset_path: Path to the dataset directory.
    :param calibration_size: Number of samples to include in the calibration set.
    :param transform: Transformations to be applied to the dataset samples.
    :param batch_size: Batch size for the DataLoader.
    :return: DataLoader for the calibration set.
    """
    dataset = datasets.ImageFolder(root=dataset_path, transform=transform)

    indices = np.random.choice(len(dataset), size=calibration_size, replace=False)
    calibration_dataset = Subset(dataset, indices)

    # DataLoader 생성
    calibration_loader = DataLoader(calibration_dataset, batch_size=batch_size, shuffle=False)

    return calibration_loader

#transform = transforms.Compose([
#    transforms.Resize((224, 224)),
#    transforms.ToTensor(),
#    # 기타 필요한 변환들
#])

#calibration_loader = sample_calibration_set(dataset_path='path/to/imagenet', 
#                                            calibration_size=1024, 
#                                            transform=transform)