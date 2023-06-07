"""
Contains functionality for creating PyTorch DataLoaders for 
image classification data.
"""
import os

import torch
import torchvision.io as io
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from .datasets import ImageFolderOneStor

import warnings

warnings.filterwarnings("ignore")

# resolve this by using _init_.py instead
#################################################
import sys
sys.path.insert(0, '../../../third_party/oneStorage/src/python/build/')
from _pywrap_oneFile import oneFile
#################################################

from PIL import Image

#NUM_WORKERS = os.cpu_count()
NUM_WORKERS = 0

def onestor_loader(path: str) -> torch.Tensor:
    #return io.read_image(path, io.image.ImageReadMode.RGB).type(torch.float32)
    #print(path)
    data = oneFile().read(path)
    #print(data)
    img_tensor = torch.frombuffer(data, dtype=torch.uint8)
    return io.decode_image(img_tensor, io.image.ImageReadMode.RGB).type(torch.float32)

#def torch_loader(path: str) -> Image.Image:
def torch_loader(path: str) -> torch.Tensor:
    transform = transforms.Compose([ transforms.PILToTensor()])
    with open(path, "rb") as f:
        img = Image.open(f)
        return transform(img.convert("RGB")).type(torch.float32)

def get_torch_dataloaders(
    train_dir: str, 
    test_dir: str, 
    transform: transforms.Compose, 
    batch_size: int, 
    num_workers: int=NUM_WORKERS
):
  """Creates training and testing DataLoaders.

  Takes in a training directory and testing directory path and turns
  them into PyTorch Datasets and then into PyTorch DataLoaders.

  Args:
    train_dir: Path to training directory.
    test_dir: Path to testing directory.
    transform: torchvision transforms to perform on training and testing data.
    batch_size: Number of samples per batch in each of the DataLoaders.
    num_workers: An integer for number of workers per DataLoader.

  Returns:
    A tuple of (train_dataloader, test_dataloader, class_names).
    Where class_names is a list of the target classes.
    Example usage:
      train_dataloader, test_dataloader, class_names = \
        = create_dataloaders(train_dir=path/to/train_dir,
                             test_dir=path/to/test_dir,
                             transform=some_transform,
                             batch_size=32,
                             num_workers=4)
  """
  # Use ImageFolder to create dataset(s)
  #train_data = datasets.ImageFolder(train_dir, transform=transform, loader = torch_loader)
  #test_data  = datasets.ImageFolder(test_dir,  transform=transform, loader = torch_loader)

  train_data = datasets.ImageFolder(train_dir, transform=transform)
  test_data  = datasets.ImageFolder(test_dir,  transform=transform)

  # Get class names
  class_names = train_data.classes

  # Turn images into data loaders
  train_dataloader = DataLoader(
      train_data,
      batch_size=batch_size,
      shuffle=True,
      num_workers=num_workers
  )
  test_dataloader = DataLoader(
      test_data,
      batch_size=batch_size,
      shuffle=True,
      num_workers=num_workers
  )

  return train_dataloader, test_dataloader, class_names


def get_onestor_dataloaders(
    train_dir: str, 
    test_dir: str, 
    transform: transforms.Compose, 
    batch_size: int, 
    num_workers: int=NUM_WORKERS
):
  """Creates training and testing DataLoaders.

  Takes in a training directory and testing directory path and turns
  them into PyTorch Datasets and then into PyTorch DataLoaders.

  Args:
    train_dir: Path to training directory.
    test_dir: Path to testing directory.
    transform: torchvision transforms to perform on training and testing data.
    batch_size: Number of samples per batch in each of the DataLoaders.
    num_workers: An integer for number of workers per DataLoader.

  Returns:
    A tuple of (train_dataloader, test_dataloader, class_names).
    Where class_names is a list of the target classes.
    Example usage:
      train_dataloader, test_dataloader, class_names = \
        = create_dataloaders(train_dir=path/to/train_dir,
                             test_dir=path/to/test_dir,
                             transform=some_transform,
                             batch_size=32,
                             num_workers=4)
  """
  # Use ImageFolder to create dataset(s)
  train_data = ImageFolderOneStor(train_dir, transform=transform)
  test_data  = ImageFolderOneStor(test_dir,  transform=transform)

  # Get class names
  class_names = train_data.classes

  # Turn images into data loaders
  train_dataloader = DataLoader(
      train_data,
      batch_size=batch_size,
      shuffle=True,
      num_workers=num_workers
  )
  test_dataloader = DataLoader(
      test_data,
      batch_size=batch_size,
      shuffle=True,
      num_workers=num_workers
  )

  return train_dataloader, test_dataloader, class_names