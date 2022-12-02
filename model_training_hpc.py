# %%
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
# import pandas as pd
# import matplotlib.pyplot as plt

import cv2

#%%
# Data Loader
import sys
# path = os.getcwd()
#sys.path.insert(0, '/Users/pauli/Documents/Studium/Master/3. Semester Auslandssemester DTU/Deep Learning/Final Project/Otovo/')
#from autotetris.dataloader import RoofDataSet
from lib.dataloader import RoofDataSet, Transforms
from lib.modeltraining import Resnet18, Resnet50, VarMSEloss, VarDiffloss, train_model, test_model
# from model_resnet_test import ResNet
from individual_tests.jaime.testing_resnets import ResNet

# %% [markdown]
# ## Import path

# %%
path = 'C:/Users/guzma/OneDrive/Documents/TEC/DTU/02456/Project/Github_Project/Dataset/data_2022-11-01/meta_data.hdf'
# path = 'data_2022-11-01/meta_data.hdf'
input_path = path

# %%
max_size = 30
dataset = RoofDataSet(path, transform=Transforms(new_size=(224,224)), mode = "constant", max_size=max_size)
imp_path = dataset.image_paths +  "/"+dataset.id[0]+"-b15-otovowms.jpeg"
image = cv2.imread(imp_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#%%
# split the dataset into training, validation and test sets
# Create testset
len_test_set = int(0.1*len(dataset))
len_train_set = len(dataset) - len_test_set

# %%
train_dataset , test_dataset  = torch.utils.data.random_split(dataset, [len_train_set, len_test_set], generator=torch.Generator().manual_seed(1))


len_valid_set = int(0.1*len(train_dataset))
len_train_set = len(train_dataset) - len_valid_set

train_dataset, valid_dataset = torch.utils.data.random_split(train_dataset, [len_train_set, len_valid_set], generator=torch.Generator().manual_seed(1))

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
# network = Resnet18()
network = Resnet50(num_classes=max_size*2) # Because our padded values are added on both sides of the image.
network.to(device)
# print(network)

# Adjust network parameter
criterion = VarDiffloss()
# optimizer = optim.SGD(network.parameters(), lr=0.001)
optimizer = optim.Adam(network.parameters(), lr=0.0001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

loss_min = 0.0001
num_epochs = 10

# Train model
model = train_model(network, criterion, optimizer, num_epochs, train_loader, valid_loader, device)

torch.save(model, 'resnet50_2_diff_1_02_12_22_j.pt')
