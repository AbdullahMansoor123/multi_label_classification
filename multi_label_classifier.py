from __future__ import print_function, division
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch.nn as nn
import torch.optim as optim
from multi_label_dataset import multi_label_dataset
import numpy as np

# preprocessing data
transform = transforms.Compose([
    transforms.Resize(255),
    # transforms.CenterCrop(224),
    # transforms.RandomHorizontalFlip(0.5),
    # transforms.AutoAugment(),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])
data_dir = 'household'
dataset = multi_label_dataset(csv_file='multi_label.csv',
                              data_dir=data_dir,
                              transform=transform)
dataset_len = len(dataset)

# train and test split
train_len, test_len = dataset_len - 100, 100
# train_set, test_set,misc_set = torch.utils.data.random_split(dataset, [200, 50, 250])# 100 represent the size of testset we want
train_set, test_set = torch.utils.data.random_split(dataset, [train_len, test_len])

# train and test split
train_len = len(dataset)
# print(len(transformed_data))

# hyper-parameters
num_classes = 5
batch_size = 20
learning_rate = 0.001

# train and test dataloader
train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=True)
# print(train_loader)

# set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('{} in use'.format(device))

# CNN Module

# pre-trained Model
model = torchvision.models.resnet50()
for param in model.parameters():
    param.requires_grad = False

model.fc = torch.nn.Linear(in_features=2048, out_features=5, bias=True)

model.to(device)

# loss and optimizer
loss_fn = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

num_epochs = 2
# training loop
for epoch in range(num_epochs):
    print(f'epoch:{epoch + 1}/{num_epochs}...........\n')
    total_correct, running_loss = 0.0, 0.0
    total = 0
    for batch, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        total += labels.size(0)*num_classes
        # Forward Pass
        outputs = model(images)
        # backward
        optimizer.zero_grad()
        loss = loss_fn(outputs, labels)
        running_loss += loss.item() * images.size(0) ### todo
        loss.backward()
        # Optimizing
        optimizer.step()
        outputs = (torch.sigmoid(outputs).round()).detach().cpu().numpy()
        labels = labels.detach().cpu().numpy()
        total_correct += (labels == outputs).sum()
        # acc_per_class = (np.mean(correct, axis=0)).round()
    print(f'train loss: {(running_loss / total):.4f} | train accuracy: {(total_correct/total):.4f}')
    # print(f'train loss: {(running_loss / total):.4f} ')

    # test loop
    with torch.no_grad():
        model.eval()  # notify our layer we are in evaluation mode
        running_loss, total_correct = 0.0, 0.0
        total = 0.0
        for batch, (images, labels) in enumerate(test_loader):
            images = images.to(device)
            labels = labels.to(device)
            total += labels.size(0) * num_classes
            outputs = model(images)
            test_loss = loss_fn(outputs, labels)
            running_loss += test_loss.item() * images.size(0)
            # prepare for test overall accuracy
            outputs = (torch.sigmoid(outputs).round()).detach().cpu().numpy()
            labels = labels.detach().cpu().numpy()
            total_correct += (labels == outputs).sum()
        # print(acc_per_class)
    print(f'test loss: {(running_loss / total):.4f} | accuracy per class: {total_correct/total}')
print('Training and testing completed!')

# save model
torch.save(model.state_dict(), 'test_classifier.pth')
