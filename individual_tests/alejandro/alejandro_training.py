import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader

import sys
sys.path.insert(0, "/Users/alex_christlieb/Documents/Ecole Polytechnique/Exchange/DTU/Courses/Deep Learning/autotetris")
# Data Loader
from lib.modeltraining import Resnet18_GAP, VarDiffloss, train_model
from lib.dataloader import RoofDataSet, Transforms

path_alejandro = '/Users/alex_christlieb/Downloads/Dataset/data_2022-11-01/meta_data.hdf'
#%%
dataset = RoofDataSet(path_alejandro, transform=Transforms(new_size=(224,224)), mode = "constant")
#%%
imp_path = dataset.image_paths +  "/"+dataset.id[0]+"-b15-otovowms.jpeg"

#%%
# split the dataset into training, validation and test sets
# Create testset
len_test_set = int(0.1*len(dataset))
len_train_set = len(dataset) - len_test_set

train_dataset , test_dataset  = torch.utils.data.random_split(dataset, [len_train_set, len_test_set], generator=torch.Generator().manual_seed(1))

len_valid_set = int(0.1*len(train_dataset))
len_train_set = len(train_dataset) - len_valid_set

train_dataset, valid_dataset = torch.utils.data.random_split(train_dataset, [len_train_set, len_valid_set])


# %%
print("The length of Train set is {}".format(len_train_set))
print("The length of Valid set is {}".format(len_valid_set))
print("The length of Test set is {}".format(len_test_set))


# shuffle and batch the datasets
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=4, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=lambda x: tuple(x_.to("cpu") for x_ in default_collate(x)))
# test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
#%%
image, centroid = next(iter(train_loader))
# print(image.shape, centroid.shape, centroid)
#%%
# network = Resnet50(num_classes=max_size*2)
network = Resnet18_GAP(num_classes=dataset.max_num_panels*2)
network.to(device)
# print(network)

# Adjust network parameter
criterion = VarDiffloss()
# SGD diverges on our model
# optimizer = optim.SGD(network.parameters(), lr=0.0001)
optimizer = optim.Adam(network.parameters(), lr=0.0001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

loss_min = 0.001
num_epochs = 5

# Train model
model = train_model(network, criterion, optimizer, num_epochs, train_loader, valid_loader, device)

torch.save(model, 'resnet_gap_5_05_12_22_a.pt')


