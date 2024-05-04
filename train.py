import os
import torch
import torchvision
import torch.optim as optim
import torch.backends.cudnn as cudnn
import numpy as np
import matplotlib.pyplot as plt

from torch import nn
from PIL import Image
from torch.optim import lr_scheduler
from torchvision.transforms import ToTensor
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, random_split

from classes import MinecraftImageClassification

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def get_datasets(root_dir_train='frames_biomes/train', root_dir_test='frames_biomes_test', transform=None):
    train_dataset = MinecraftImageClassification(root_dir=root_dir_train, transform=transform)
    test_dataset = MinecraftImageClassification(root_dir=root_dir_test, transform=transform)
    return train_dataset, test_dataset

def show_images(dataset):
    num_classes = len(dataset.classes)
    fig, axs = plt.subplots(1, num_classes, figsize=(3*num_classes, 3), constrained_layout=True)
    for i, dataset_class in enumerate(dataset.classes):
        class_idx = dataset.class_indices.index(i)
        image, _ = dataset[class_idx]
        # if the dataset has 'transform' attribute and it is not None
        if getattr(dataset, "transform") and dataset.transform is not None:
            # unnormalize images for display
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            image = std * image.permute(1,2,0).numpy() + mean
            image = np.clip(image, 0, 1)
        
        axs[i].imshow(image)
        axs[i].set_title(dataset_class)
        axs[i].axis('off')
    plt.show()

def main():
    # transformations for the training images based on ResNet normalization requirements https://pytorch.org/hub/pytorch_vision_resnet/
    transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset, test_dataset = get_datasets(transform=transform)

    show_images(train_dataset)

    print('Train dataset size: ', len(train_dataset))
    print('Test dataset size: ', len(test_dataset))
    print('Number of classes: ', len(train_dataset.classes))

    val_size = int(0.2 * len(train_dataset))
    train_size = len(train_dataset) - val_size

    # ensure the randomness of the splitting process different every time you run the code
    torch.manual_seed(0) 

    train_dataset_split, val_dataset = random_split(train_dataset, [train_size, val_size])

    batch_size = 8

    train_loader = DataLoader(train_dataset_split, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    print('Train dataloader size: ', len(train_loader))
    print('Validation dataloader size: ', len(val_loader))
    print('Test dataloader size: ', len(test_loader))