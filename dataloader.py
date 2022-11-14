import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

import cv2
import matplotlib.pyplot as plt
from skimage import io


class MyDataset(Dataset):
  def __init__(self,file_name, transform=None):
    img_df=pd.read_hdf(file_name, '/d')
    img_name=img_df.iloc[:,0] #.str.split('-b15',n = 1, expand = True)[0].tolist()
    centroid=img_df.iloc[:,6].values
    
    self.surface = img_df.iloc[:,4].values
    self.base = img_df.iloc[:,3].values
    self.id = img_name

    self.transform = transform
    self.image_paths = [file_name.split('/meta')[0]+'/'+name+'-b15-otovowms.jpeg' for name in img_name]
    self.label = centroid
 
    # self.x_train=torch.tensor(x,dtype=torch.float32)
    # self.y_train=torch.tensor(y,dtype=torch.float32)
 
  def __len__(self):
    return len(self.label.shape[0])
   
  def __getitem__(self,idx):
    image_filepath = self.image_paths[idx]
    image = cv2.imread(image_filepath)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    label = self.label[idx]
    if self.transform is not None:
        image = self.transform(image=image)["image"]
    
    return image, label, self.id

  def show_centroids(self, idx):
    """Show image with centroids"""
    label = np.array(self.label[idx])
    # var2 = np.array(self.surface[idx])
    var3 = np.array(self.base[idx])
    plt.title(self.id.iloc[idx])
    print(self.id.iloc[idx])
    plt.imshow(io.imread(self.image_paths[idx]))
    plt.scatter(var[:,1], var[:,0], s=10, marker='.', c='r')
    # plt.plot(var2[:,1], var2[:,0], c='k')
    plt.plot(var3[:,1], var3[:,0], c='b')
    plt.plot(var)    # plt.xticks([])
    # plt.yticks([])
    plt.pause(0.001)  # pause a bit so that plots are updated
    plt.show() 