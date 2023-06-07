import torch
############# code changes ###############
import intel_extension_for_pytorch as ipex
############# code changes ###############

from torchvision import datasets, models, transforms
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim

from onestor.image_classification import train
from onestor.image_classification import get_torch_dataloaders, get_onestor_dataloaders
from onestor.image_classification import save_model, print_train_time

import argparse

#import torchvision
#torchvision.set_image_backend('accimage')

def training(args):
    print(args)

    num_epochs = args.epochs
    num_workers = args.num_workers
    rand_seed  = args.seed
    batch_size = args.batch_size

    train_dir  = args.train_dir
    test_dir   = args.val_dir
    model_dir  = args.model_dir
    model_name = args.model_name
    enable_onstor = args.onestor

    # Setup target device
    device = torch.device("xpu" if ipex.xpu.is_available() else "cpu")
    print(device)

    # Start the timer
    from timeit import default_timer as timer 

    # Create transforms
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])

    if enable_onstor:
        data_transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.RandomAffine(0, shear=10, scale=(0.8,1.2)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
            ])
    else:
        data_transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.RandomAffine(0, shear=10, scale=(0.8,1.2)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
            ])

    # Create DataLoaders
    if enable_onstor:    
        train_dataloader, test_dataloader, class_names = get_onestor_dataloaders(
            train_dir   = train_dir,
            test_dir    = test_dir,
            transform   = data_transform,
            batch_size  = batch_size,
            num_workers = num_workers
        )
    else:
        train_dataloader, test_dataloader, class_names = get_torch_dataloaders(
            train_dir   = train_dir,
            test_dir    = test_dir,
            transform   = data_transform,
            batch_size  = batch_size,
            num_workers = num_workers
        )

    model = models.resnet50(pretrained=True).to(device)

    # Freeze all base layers in the "features" section of the model
    for param in model.parameters():
        param.requires_grad = False   

    # Replace model last layer with linear layer
    model.fc = nn.Sequential(
                nn.Linear(2048, 128),
                nn.ReLU(inplace=True),
                nn.Linear(128, 2)).to(device)

    # Set loss and optimizer
    loss_fn   = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters())

    start_time = timer()

    train(model = model,
        train_dataloader = train_dataloader,
        test_dataloader  = test_dataloader,
        optimizer        = optimizer,
        loss_fn          = loss_fn,
        epochs           = num_epochs,
        device           = device)

    # End the timer and print out how long it took
    end_time = timer()

    print_train_time(start_time, end_time, device)

    # Save the model with help from utils.py
    save_model(model = model,
                target_dir = model_dir,
                model_name = model_name)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--batch_size', type=int, default=32, metavar='N',
                        help='input batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default=3, metavar='N',
                        help='number of epochs to train (default: 3)')
    parser.add_argument('--num_workers', type=int, default=0, metavar='N',
                        help='number of workers (default: 0)')                        
    parser.add_argument('--seed', type=int, default=42, metavar='S',
                        help='random seed (default: 42)')

#    parser.add_argument('--train_dir', type=str, default="/mnt/data2/data/datasets/pizza_steak_sushi/train",
#                        help='training dataset folder')
#    parser.add_argument('--val_dir',   type=str, default="/mnt/data2/data/datasets/pizza_steak_sushi/test",
#                        help='validation dataset folder')
#    parser.add_argument('--model_dir', type=str, default="/mnt/data2/data/datasets/pizza_steak_sushi/model",
#                        help='model save folder')

    parser.add_argument('--train_dir', type=str, default="/mnt/data2/data/datasets/imagenet-object-localization-challenge/ILSVRC/Data/CLS-LOC/train",
                        help='training dataset folder')
    parser.add_argument('--val_dir',   type=str, default="/mnt/data2/data/datasets/imagenet-object-localization-challenge/ILSVRC/Data/CLS-LOC/test",
                        help='validation dataset folder')
    parser.add_argument('--model_dir', type=str, default="/mnt/data2/data/datasets/imagenet-object-localization-challenge/ILSVRC/Data/CLS-LOC/model",
                        help='model save folder')

    parser.add_argument('--model_name', type=str, default="saved_model.pth",
                        help='model save name')                        
    parser.add_argument('--onestor', action='store_true',
                        help='Enable/Disable oneStorage dataloader')

    training(parser.parse_args())