import torch
import torchvision
############# code changes ###############
import intel_extension_for_pytorch as ipex
############# code changes ###############

LR = 0.001
#DOWNLOAD = True
DATA = 'datasets/cifar10/'

transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((224, 224)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

################## code changes ####################
train_dataset = ipex.oneFileIterableDataset(root = DATA)
################## code changes ####################

print("total number of files:{total_files}".format(total_files = len(train_dataset)))

##### code changes ################################

train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=4
)

model = torchvision.models.resnet50()
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr = LR, momentum=0.9)
model.train()
#################################### code changes ################################
model = model.to("xpu")
model, optimizer = ipex.optimize(model, optimizer=optimizer, dtype=torch.float32)
#################################### code changes ################################

for batch_idx, (data, target) in enumerate(train_loader):
    print(batch_idx)
    print(data)
    #print(target[0:3])

#for batch_idx, (data, target) in enumerate(train_loader):
    ########## code changes ##########
#    data = data.to("xpu")
#    target = target.to("xpu")
    ########## code changes ##########
#    optimizer.zero_grad()
#    output = model(data)
#    loss = criterion(output, target)
#    loss.backward()
#    optimizer.step()
#    print(batch_idx)
#torch.save({
#     'model_state_dict': model.state_dict(),
#     'optimizer_state_dict': optimizer.state_dict(),
#     }, 'checkpoint.pth')