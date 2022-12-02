import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from lib.dataloader import RoofDataSet
from lib.dataloader import Transforms
from lib.modeltraining import Resnet18, GAP_Model, train_model, VarDiffloss
#%%
# Github_Project\Dataset\data_2022-11-01\meta_data.hdf
# C:\Users\guzma\OneDrive\Documents\TEC\DTU\02456\Project\Github_Project\Dataset\data_2022-11-01\meta_data.hdf
# path = '/Users/pauli/Documents/Studium/Master/3. Semester Auslandssemester DTU/Deep Learning/Final Project/Otovo/data_full/meta_data.hdf'
# path = './data_updated/meta_data.hdf'
path_alejandro = "/Users/alex_christlieb/Desktop/small_sample_out/metadata_sample.hdf"


#df = pd.read_hdf(path, '/d')
# centroid=df.iloc[:,6].values
# test = df.building_id.str.split('-b15',n = 1, expand = True)[0].tolist()
#centroid = np.array(df.panel_centroids.to_list())
#%%
dataset = RoofDataSet(path_alejandro, transform=Transforms(new_size=(224,224)), mode = "constant")
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

print("The length of Train set is {}".format(len_train_set))
print("The length of Valid set is {}".format(len_valid_set))
print("The length of Test set is {}".format(len_test_set))


# shuffle and batch the datasets
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=4, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
#%%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
#%%
image, centroid = next(iter(train_loader))
print(image.shape, centroid.shape, centroid)
#%%
network = GAP_Model()
network.to(device)
# print(network)

# Adjust network parameter
criterion = VarDiffloss()
optimizer = optim.Adam(network.parameters(), lr=0.0001)

loss_min = np.inf
num_epochs = 10

# Train model
model = train_model(network, criterion, optimizer, num_epochs, train_loader, valid_loader, device)

torch.save(model, 'resnet_first_28_11_22_j')




