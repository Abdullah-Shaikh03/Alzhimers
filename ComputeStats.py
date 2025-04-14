import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    DATASET_PATH = './Dataset'

    transform = transforms.Compose([
        transforms.Grayscale(),  # Optional
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])

    dataset = datasets.ImageFolder(root=DATASET_PATH, transform=transform)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4)

    n_images = 0
    sum_ = 0.0
    sum_squared = 0.0

    print("Calculating mean and std...")

    for images, _ in dataloader:
        images = images.to(device)
        n = images.size(0)
        n_images += n
        sum_ += images.sum().item()
        sum_squared += (images ** 2).sum().item()

    mean = sum_ / (n_images * 128 * 128)
    std = (sum_squared / (n_images * 128 * 128) - mean ** 2) ** 0.5

    print(f"Mean: {mean:.6f}")
    print(f"Std: {std:.6f}")

if __name__ == "__main__":
    main()
