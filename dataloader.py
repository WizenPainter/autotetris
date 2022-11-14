import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

import cv2
import matplotlib.pyplot as plt
from skimage import io
from torchvision.transforms import ToTensor


class MyDataset(Dataset):
    def __init__(self, file_name, transform=None):
        """
        Args : 
            file_name (string) :  path to the metadata file
            """
        self.transform = transform 
        img_df=pd.read_hdf(file_name, '/d') #Read metadata 
        img_df["number_panels"] = img_df["panel_centroids"].apply(lambda x: len(x)) #Compute number of panels per building 
        
        self.max_num_panels = img_df["number_panels"].max() #store the maximum number of panels 
        self.id = img_df["building_id"]  #store necessary data
        # self.polygons = img_df["panel_polygons"] 

        img_df["centroids_pad"] = img_df["panel_centroids"].apply(self.__pad_centroids)
        self.image_paths = [file_name.split('/meta')[0]+'/'+name+'-b15-otovowms.jpeg' for name in self.id]
        self.centroid = img_df["centroids_pad"].values #Get centroids 
    
        # self.x_train=torch.tensor(x,dtype=torch.float32)
        # self.y_train=torch.tensor(y,dtype=torch.float32)

    def __pad_centroids(self, x):
        """Pad x with 'fill_values' to standardize length equal to the maximum number of centroids"""
        fill_values = np.array([0,0])
        return np.apply_along_axis(lambda x: np.pad(np.array(x), (0,self.max_num_panels-len(x)), mode = 'constant', constant_values = fill_values), axis = 0, arr = x)

    def __len__(self):
        """Get dataset size"""
        return len(self.centroid.shape[0])
    
    def __getitem__(self,idx):
        """Get one item of the dataset"""
        image_filepath = self.image_paths[idx]
        image = cv2.imread(image_filepath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.transform is not None:
            image = self.transform(image=image)["image"]
        
        return image, self.centroid[idx]

    def centroid_padding():
        """Padd all labels to obtain the same size """

    def show_centroids(self, idx):
        """Show image with centroids"""
        centroids = np.array(self.centroid[idx])

        plt.title(self.id.iloc[idx])
        print(self.id.iloc[idx])
        plt.scatter(centroids[:, 1], centroids[:, 0], s=10, marker='.', c='k')
        plt.imshow(io.imread(self.image_paths[idx]))

        # plt.xticks([])
        # plt.yticks([])
        plt.pause(0.001)  # pause a bit so that plots are updated
        plt.show() 