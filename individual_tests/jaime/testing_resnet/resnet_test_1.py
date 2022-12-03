import torch
import torchvision.models as models
from torch import nn
from lib.dataloader import RoofDataSet
from lib.dataloader import Transforms
from torch.utils.data import DataLoader

import cv2
from torch.utils.data.dataloader import default_collate

from lib.modeltraining import VarDiffloss, train_model

# Create a multi-input model
class SolarPanelPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet50 = models.resnet50(pretrained=False)
        
        # Replace final layer of ResNet-50 with a new fully-connected layer
        num_ftrs = self.resnet50.fc.in_features
        self.resnet50.fc = nn.Linear(num_ftrs, 256)
        
        # Add additional layers for coordinate input
        self.coord_embedding = nn.Linear(2, 64)
        self.coord_bn = nn.BatchNorm1d(64)
        self.fc = nn.Linear(256 + 64, 1)
        
        # Add dropout layer to prevent overfitting
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, img, coords):
        # Extract features from image using ResNet-50
        features = self.resnet50(img)
        
        # Embed coordinates and apply batch normalization
        coords_embedded = self.coord_embedding(coords)
        coords_bn = self.coord_bn(coords_embedded)
        
        # Concatenate features and embedded coordinates
        concat = torch.cat((features, coords_bn), dim=1)
        
        # Pass concatenated features through dropout and fully-connected layer to produce predictions
        output = self.dropout(concat)
        output = self.fc(output)
        
        return output

path = 'C:/Users/guzma/OneDrive/Documents/TEC/DTU/02456/Project/Github_Project/Dataset/data_2022-11-01/meta_data.hdf'

dataset = RoofDataSet(path, transform=Transforms(new_size=(224,224)), mode = "constant") # Optimal size is 224 according to OpenAI
imp_path = dataset.image_paths +  "/"+dataset.id[0]+"-b15-otovowms.jpeg"
image = cv2.imread(imp_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

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
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=4, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=lambda x: tuple(x_.to("cpu") for x_ in default_collate(x)))

# Create model and move to device (e.g. GPU)
model = SolarPanelPredictor()
model = model.to(device)

# Define loss function and optimizer
criterion = nn.BCEWithLogitsLoss()
# criterion = VarDiffloss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)


loss_min = 0.001
num_epochs = 5 # after 5 epochs the model has almost no improvement that justifies the time spent.

# Train model
model = train_model(model, criterion, optimizer, num_epochs, train_loader, valid_loader, device)

torch.save(model, 'resnet50_auto_diff_adam_10_02_12_22_j-v2.pt')


# # Train model on dataset
# for inputs, labels in dataset:
#     # Move data to device
#     inputs = inputs.to(device)
#     labels = labels.to(device)
    
#     # Forward pass
#     outputs = model(inputs, labels)
#     loss = criterion(outputs, labels)
    
#     # Backward pass and optimization step
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()
    
# # Use model to make predictions on new images
# new_imgs = ... # Load new images
# new_coords = torch.zeros((len(new_imgs), 2)).to(device) # Empty tensor for coordinates

# with torch.no_grad():
#     outputs = model(new_imgs, new_coords)
#     predictions = (outputs > 0).long() # Convert logits to binary predictions
