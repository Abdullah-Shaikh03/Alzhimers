# DataLoader.py
import os
import yaml
# import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import nibabel as nib
from PIL import Image
# import pandas as pd
# from pathlib import Path


class MedicalImageDataset(Dataset):
    """Dataset for loading medical brain MRI images with classifications:
    AD (Alzheimer's Disease), CN (Cognitively Normal), 
    EMCI (Early Mild Cognitive Impairment), LMCI (Late Mild Cognitive Impairment)
    """
    
    def __init__(self, root_dir, transform=None, split='train'):
        """
        Args:
            root_dir (str): Directory with all the images organized in folders by class
            transform (callable, optional): Optional transform to be applied on a sample
            split (str): 'train', 'val', or 'test'
        """
        self.root_dir = root_dir
        self.transform = transform
        self.split = split
        
        # Map class names to numeric labels
        self.class_to_idx = {
            'CN': 0,    # Cognitively Normal (control)
            'EMCI': 1,  # Early Mild Cognitive Impairment
            'LMCI': 2,  # Late Mild Cognitive Impairment
            'AD': 3     # Alzheimer's Disease
        }
        
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
        
        # Load all image paths and labels
        self.samples = self._load_dataset()
        
    def _load_dataset(self):
        """Load all image paths and corresponding labels"""
        samples = []
        
        # Traverse through each class directory
        for class_name in self.class_to_idx.keys():
            class_dir = os.path.join(self.root_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
                
            # Get all files in the class directory
            all_files = [f for f in os.listdir(class_dir) 
                        if os.path.isfile(os.path.join(class_dir, f)) and 
                        (f.endswith('.nii') or f.endswith('.nii.gz') or f.endswith('.png') or f.endswith('.jpg'))]
            
            # Split files according to train/val/test (70/15/15 split by default)
            np.random.seed(42)  # For reproducibility
            np.random.shuffle(all_files)
            
            if self.split == 'train':
                files = all_files[:int(0.7 * len(all_files))]
            elif self.split == 'val':
                files = all_files[int(0.7 * len(all_files)):int(0.85 * len(all_files))]
            else:  # test
                files = all_files[int(0.85 * len(all_files)):]
            
            for file_name in files:
                file_path = os.path.join(class_dir, file_name)
                label = self.class_to_idx[class_name]
                samples.append((file_path, label))
        
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        # Load image based on file extension
        if img_path.endswith('.nii') or img_path.endswith('.nii.gz'):
            # Load NIfTI file
            nii_img = nib.load(img_path)
            # Get data as numpy array
            img_data = nii_img.get_fdata()
            
            # Extract a middle slice (for 3D volumes)
            if len(img_data.shape) == 3:
                middle_slice = img_data.shape[2] // 2
                img_data = img_data[:, :, middle_slice]
            
            # Normalize to [0, 255]
            img_data = ((img_data - img_data.min()) / (img_data.max() - img_data.min()) * 255).astype(np.uint8)
            
            # Convert to PIL Image
            image = Image.fromarray(img_data)
        else:
            # Regular image formats (PNG, JPG)
            image = Image.open(img_path).convert('L')  # Convert to grayscale
        
        if self.transform:
            image = self.transform(image)
            
        return image, label


def get_transforms(image_size=128):
    """Get image transformations for training and validation/testing"""
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])  # Normalize to [-1, 1]
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])  # Normalize to [-1, 1]
    ])
    
    return train_transform, val_transform


def get_dataloaders(config):
    """Create and return dataloaders based on configuration"""
    train_transform, val_transform = get_transforms(config['image_size'])
    
    # Create datasets
    train_dataset = MedicalImageDataset(
        root_dir=config['dataset_path'],
        transform=train_transform,
        split='train'
    )
    
    val_dataset = MedicalImageDataset(
        root_dir=config['dataset_path'],
        transform=val_transform,
        split='val'
    )
    
    test_dataset = MedicalImageDataset(
        root_dir=config['dataset_path'],
        transform=val_transform,
        split='test'
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=True
    )
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")
    
    return train_loader, val_loader, test_loader


# Example usage
if __name__ == "__main__":
    # Load config
    with open("Config.yaml", 'r') as file:
        config = yaml.safe_load(file)
    
    train_loader, val_loader, test_loader = get_dataloaders(config)
    
    # Print some information
    for images, labels in train_loader:
        print(f"Batch shape: {images.shape}")
        print(f"Labels: {labels}")
        break