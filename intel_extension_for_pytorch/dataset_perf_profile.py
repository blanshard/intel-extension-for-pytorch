import torch
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
from torchvision.transforms.autoaugment import AutoAugmentPolicy
from tensorboardX import SummaryWriter

from time import time

############# code changes ###############
import intel_extension_for_pytorch as ipex
from onestor.image_classification.datasets import ImageFolderOneStor
############# code changes ###############

# Setup target device
device = torch.device("xpu" if ipex.xpu.is_available() else "cpu")
print(device)

# Set hyperparameters
num_epochs = 10
batch_size = 64
learning_rate = 0.01

transform = transforms.Compose([
    transforms.Resize(96),
    transforms.CenterCrop(80),
    #transforms.AutoAugment(AutoAugmentPolicy.IMAGENET),
    transforms.ToTensor(),
    #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load the ImageNet Object Localization Challenge dataset
#train_dataset = torchvision.datasets.ImageFolder(root='/mnt/data2/data/datasets/imagenet-object-localization-challenge/ILSVRC/Data/CLS-LOC/train', transform=transform)

 # Use ImageFolder to create dataset(s)
train_dataset = ImageFolderOneStor('/mnt/data2/data/datasets/imagenet-object-localization-challenge/ILSVRC/Data/CLS-LOC/train', transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=6)

# Train the model...
print(len(train_loader))

start = time()
for batch_idx, batch in enumerate(train_loader):
    print(batch_idx)
    #if batch_idx == 1:
    #    break

print("Dataset looping time: {:.3f} seconds".format((time() - start)))
print('Finished Dataset read')