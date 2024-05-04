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

from torchvision import models
from torchvision.models.resnet import ResNet18_Weights
from torchvision.models.resnet import ResNet152_Weights
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

class MinecraftImageClassification(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.classes = os.listdir(self.root_dir)
        self.filenames = []
        self.class_indices = []

        for class_index, dataset_class in enumerate(self.classes):
            class_dir = os.path.join(self.root_dir, dataset_class)
            class_filenames = os.listdir(class_dir)

            self.filenames += list(map(lambda x: os.path.join(class_dir, x), class_filenames))
            self.class_indices += [class_index]*len(class_filenames)

        self.transform = transform

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        image = Image.open(self.filenames[idx])
        class_index = self.class_indices[idx]

        if self.transform:
            image = self.transform(image)

        return image, class_index


class Model:

def train_model(model, criterion, optimizer, scheduler, num_epochs=10):
    best_model_wts = model.state_dict()
    best_acc = 0.0

    for epoch in range(num_epochs):
        print()
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                dataloader = train_loader
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                dataloader = val_loader
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in tqdm(dataloader):
                inputs, labels = inputs.to(device), labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloader.dataset)
            epoch_acc = running_corrects.double() / len(dataloader.dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()

    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

def test_model(model):
    model.eval()   # Set model to evaluation mode

    running_corrects = 0

    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        with torch.no_grad():
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
                        
        running_corrects += torch.sum(preds == labels.data)
            
    acc = running_corrects.double() / len(test_loader.dataset)

    print('Test Acc: {:.4f}'.format(acc))

    # graph 5 random test images with their predicted and true labels
    fig, axs = plt.subplots(1, num_classes, figsize=(25, 5), constrained_layout=True)
    for i in range(num_classes):
        idx = np.random.randint(0, len(test_loader.dataset))
        inputs, labels = test_loader.dataset[idx]
        inputs = inputs.unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

        # unnormalize images for display
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        inputs = std * inputs.squeeze().permute(1,2,0).cpu().numpy() + mean
        inputs = np.clip(inputs, 0, 1)

        axs[i].imshow(inputs)
        axs[i].set_title('Predicted: {}\nActual: {}'.format(test_dataset.classes[preds.item()], test_dataset.classes[labels]))
        axs[i].axis('off')

    plt.show()