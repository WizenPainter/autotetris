import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import cv2
import matplotlib.pyplot as plt
from skimage import io
from PIL import Image


class RoofDataSet(Dataset):
    def __init__(self, file_name, transform=None):
        """
        Args : 
            file_name (string) :  path to the metadata file
            transform (Transform) : object with transforms to be applied 
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
        return self.centroid.shape[0]
    
    def __getitem__(self,idx):
        """Get one item of the dataset"""
        image_filepath = self.image_paths[idx]
        image = io.imread(image_filepath)
        image = Image.fromarray(image) #store as PIL Image 
        
        if self.transform:
            image, centroids = self.transform(image, self.centroid[idx])
            #sample['centroids'] = self.transform(sample['centroids'])
        else: 
            centroids = self.centroid[idx]

        #Floats needed in pytorch models
        return image.float(), centroids.float()

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

class Transforms():
    def __init__(self, new_size = (256, 256), transform_list = ["resize", "to_tensor"]):
        """Args: 
                img_size: indicates the desired size for resizing
                transform_list: list with desired transforms to be computed. 
                    NOTE: 'transform_dic' in '__call__' must be updated when new transform functions are added            
        """
        self.new_size = new_size 
        self.transform_list = transform_list

    def to_tensor(self, image, centroids):
        """Convert image and centroids into tensor"""
        tensor = transforms.ToTensor()
        return tensor(image), torch.tensor(centroids)

    def resize(self, image, centroids):
        """Resize image and centroids into 'new_size'"""
        image = transforms.functional.resize(image, self.new_size)
        centroids = centroids * [self.new_size[0] / 500, self.new_size[1] / 500]
        return image, centroids

    def __call__(self, image, centroids):
        """Execute all desired transforms"""
        transform_dic = {"resize": self.resize, "to_tensor": self.to_tensor}
        for transform in self.transform_list:
            image, centroids = transform_dic[transform](image, centroids)
        # image, centroids = self.resize(image, centroids)
        # image, centroids = self.to_tensor(image, centroids)
        # image = transforms.functional.normalize(image, [0.5], [0.5]) #This could be contemplated 

        return image, centroids 

def show_centroids(image, centroids, tensor=False):
    """Show image with centroids"""
    plt.figure()
    if tensor:
        image = image.permute(1, 2, 0)
    plt.imshow(image)
    plt.scatter(centroids[:, 1], centroids[:, 0], s=10, marker='.', c='r')
    plt.pause(0.001)  # pause a bit so that plots are updated
    plt.show()


