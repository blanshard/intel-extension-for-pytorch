import torch
############# code changes ###############
import intel_extension_for_pytorch as ipex
############# code changes ###############

from torchvision import datasets, models, transforms
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim

from onestor.image_classification import train
from onestor.image_classification import get_torch_dataloaders
from onestor.image_classification import save_model, print_train_time

# Setup hyperparameters
NUM_EPOCHS = 10
BATCH_SIZE = 32

# Setup directories
train_dir = "./onestor/image_classification/data/train"
test_dir  = "./onestor/image_classification/data/validation"

# Setup target device
device = torch.device("xpu" if ipex.xpu.is_available() else "cpu")
print(device)

# Start the timer
from timeit import default_timer as timer 
start_time = timer()

# Create transforms
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

data_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomAffine(0, shear=10, scale=(0.8,1.2)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize
    ])

# Create DataLoaders with help from data_setup.py
train_dataloader, test_dataloader, class_names = get_torch_dataloaders(
    train_dir   = train_dir,
    test_dir    = test_dir,
    transform   = data_transform,
    batch_size  = BATCH_SIZE
)

model = models.resnet50().to(device)
    
for param in model.parameters():
    param.requires_grad = False   
    
model.fc = nn.Sequential(
               nn.Linear(2048, 128),
               nn.ReLU(inplace=True),
               nn.Linear(128, 2)).to(device)

# Set loss and optimizer
loss_fn   = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters())

train(model = model,
    train_dataloader = train_dataloader,
    test_dataloader = test_dataloader,
    optimizer = optimizer,
    loss_fn = loss_fn,
    epochs =  NUM_EPOCHS,
    device = device)

# End the timer and print out how long it took
end_time = timer()

print_train_time(start_time, end_time, device)

# Save the model with help from utils.py
save_model(model=model,
                 target_dir="models",
                 model_name="saved_model.pth")