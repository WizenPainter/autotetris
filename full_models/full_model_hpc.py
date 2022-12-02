import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import cv2

#%%
# Data Loader
import sys
# path = os.getcwd()
#sys.path.insert(0, '/Users/pauli/Documents/Studium/Master/3. Semester Auslandssemester DTU/Deep Learning/Final Project/Otovo/')
#from autotetris.dataloader import RoofDataSet
from lib.dataloader import RoofDataSet
from lib.dataloader import Transforms
from lib.modeltraining import Resnet18, model_resnet18
#%%
<<<<<<< HEAD
# Github_Project\Dataset\data_2022-11-01\meta_data.hdf
# C:\Users\guzma\OneDrive\Documents\TEC\DTU\02456\Project\Github_Project\Dataset\data_2022-11-01\meta_data.hdf
path = '/Users/pauli/Documents/Studium/Master/3. Semester Auslandssemester DTU/Deep Learning/Final Project/Otovo/data_full/meta_data.hdf'
input_path = path
=======
path = './data_updated/meta_data.hdf'
>>>>>>> 9af3b8b21c868c4882793612d6f0eeed00696ef8
print(path)
#df = pd.read_hdf(path, '/d')
# centroid=df.iloc[:,6].values
# test = df.building_id.str.split('-b15',n = 1, expand = True)[0].tolist()
#centroid = np.array(df.panel_centroids.to_list())
#%%
dataset = RoofDataSet(path, transform=Transforms(new_size=(256,256)), mode = "constant")
imp_path = dataset.image_paths +  "/"+dataset.id[0]+"-b15-otovowms.jpeg"
image = cv2.imread(imp_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#%%
# split the dataset into training, validation and test sets
# Create testset
len_test_set = int(0.1*len(dataset))
len_train_set = len(dataset) - len_test_set

train_dataset , test_dataset  = torch.utils.data.random_split(dataset, [len_train_set, len_test_set])


len_valid_set = int(0.1*len(train_dataset))
len_train_set = len(train_dataset) - len_valid_set

train_dataset, valid_dataset = torch.utils.data.random_split(train_dataset, [len_train_set, len_valid_set])

print("The length of Train set is {}".format(len_train_set))
print("The length of Valid set is {}".format(len_valid_set))
print("The length of Test set is {}".format(len_test_set))


# shuffle and batch the datasets
train_loader = DataLoader(train_dataset, batch_size=68, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=68, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
#%%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
#%%
image, centroid = next(iter(train_loader))
print(image.shape, centroid.shape, centroid)
#%%
network = Resnet18()
network.to(device)
# print(network)

# Adjust network parameter
criterion = nn.MSELoss()
optimizer = optim.Adam(network.parameters(), lr=0.001)

loss_min = np.inf
num_epochs = 50

# Train model
model = model_resnet18(network, criterion, optimizer, num_epochs, train_loader, valid_loader, device)

<<<<<<< HEAD
torch.save(model, 'resnet_first_28_11_22_j')


=======
torch.save(model, 'resnet18_constant_minus100_27_11_22.pt')
>>>>>>> 9af3b8b21c868c4882793612d6f0eeed00696ef8


