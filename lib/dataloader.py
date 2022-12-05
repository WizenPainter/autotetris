import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import cv2
import matplotlib.pyplot as plt
from skimage import io
from PIL import Image
from math import floor


class RoofDataSet(Dataset):
    def __init__(self, file_name, transform=None, mode = "wrap"):
        """
        Args : 
            file_name (string) :  path to the metadata file
            transform (Transform) : object with transforms to be applied 
            mode: one option from ["constant", "wrap"]. Wrap reinforces current centroids
            """
        print("-"*20, "Initializing dataset", "-"*20)
        self.transform = transform 
        img_df=pd.read_hdf(file_name, '/d') #Read metadata 
        print("-->", "Metadata read")

        img_df["number_panels"] = img_df["panel_centroids"].apply(lambda x: len(x)) #Compute number of panels per building 
        percentiles = img_df["number_panels"].describe(percentiles = [0.1, 0.25, 0.75, 0.9]).to_dict() #get dictionary with percentile values
        self.min_num_panels = floor(percentiles['25%'])
        self.max_num_panels = floor(percentiles['75%'])
        print("-->", "Num_panels computed")
        # self.polygons = img_df["panel_polygons"] 

        img_df = img_df[(img_df.number_panels < self.max_num_panels) & (img_df.number_panels > self.min_num_panels)] #drop samples that have too many or too few panels
        print("-->", "Samples with many panels dropped")

        img_df = img_df.reset_index(drop = True) #reset indexes to avoid empty spaces
        self.id = img_df["building_id"]  #store necessary data
        img_df["centroids_pad"] = img_df["panel_centroids"].apply(self.__pad_centroids, args = (mode,)) #apply padding to panels
        print("-->", "Padding samples")
        self.image_paths = file_name.split('/meta')[0]
        # self.image_paths = [file_name.split('/meta')[0]+'/'+name+'-b15-otovowms.jpeg' for name in self.id]
        self.centroid = img_df["centroids_pad"].values #Get centroids 
        print("-->", "Dataset ready")
    
        # self.x_train=torch.tensor(x,dtype=torch.float32)
        # self.y_train=torch.tensor(y,dtype=torch.float32)

    def __pad_centroids(self, x, mode):
        """Pad x with 'fill_values' to standardize length equal to the maximum number of centroids.
            Arguments: 
                -x: row of pandas df
        """
        options = {"constant": lambda x: np.pad(np.array(x), (0,self.max_num_panels-len(x)), mode = 'constant', constant_values = np.array([0,0])),
                    "wrap": lambda x: np.pad(np.array(x), (0,self.max_num_panels-len(x)), mode = 'wrap')}
        fill_mode = options[mode]
        return np.apply_along_axis(fill_mode, axis = 0, arr = x)

    def __len__(self):
        """Get dataset size"""
        return self.centroid.shape[0]
    
    def __getitem__(self,idx):
        """Get one item of the dataset"""
        img_id = self.id[idx]
        image_filepath = self.image_paths+ "/"+img_id+"-b15-otovowms.jpeg"
        image = io.imread(image_filepath)
        image = Image.fromarray(image) #store as PIL Image 
        
        if self.transform:
            # print(self.id[idx])
            image, centroids = self.transform(image, self.centroid[idx])
            #sample['centroids'] = self.transform(sample['centroids'])
        else: 
            centroids = self.centroid[idx]

        #Floats needed in pytorch models
        return image.float(), centroids.float() # Change is necessary to run Network loss

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
    def __init__(self, new_size = (224, 224), transform_list = ["resize", "to_tensor"], to_heatmap = False):
        """Args: 
                img_size: indicates the desired size for resizing
                transform_list: list with desired transforms to be computed. 
                    NOTE: 'transform_dic' in '__call__' must be updated when new transform functions are added            
        """
        self.new_size = new_size 
        self.transform_list = transform_list
        self.heatmap = to_heatmap

    def to_tensor(self, image, centroids):
        """Convert image and centroids into tensor"""
        tensor = transforms.ToTensor()
        return tensor(image), torch.tensor(centroids)

    def resize(self, image, centroids):
        """Resize image and centroids into 'new_size'"""
        image = transforms.functional.resize(image, self.new_size)
        try:
            centroids = centroids * (self.new_size[0] / 500, self.new_size[1] / 500)
            # centroids = centroids.dot([self.new_size[0] / 500, self.new_size[1] / 500]) # Normal multiplication cannot work, .dot is necessary for this to work
        except ValueError:
            print(centroids)
            centroids = np.transpose(centroids).dot([self.new_size[0] / 500, self.new_size[1] / 500])
        return image, centroids
    
    def to_heatmap(self, image, centroids):
        """Convert centroids into heatmap"""
        masks = []
        if type(image) != torch.Tensor:
            tensor = transforms.ToTensor()
            image = tensor(image)
        if type(centroids) == torch.Tensor:
            centroids = centroids.numpy()
            centroids = centroids[0]
        mask = torch.zeros_like(image[0])
        for coords in centroids:
            coord = [int(round(c)) for c in coords]
            mask[coord[0], coord[1]] = 1
        masks.append(mask)
        return image, torch.stack(masks)

    def __call__(self, image, centroids):
        """Execute all desired transforms"""
        transform_dic = {"resize": self.resize, "to_tensor": self.to_tensor, 'to_heatmap': self.to_heatmap}
        for transform in self.transform_list:
            image, centroids = transform_dic[transform](image, centroids)
        # image, centroids = self.resize(image, centroids)
        # image, centroids = self.to_tensor(image, centroids)
        # image = transforms.functional.normalize(image, [0.5], [0.5]) #This could be contemplated 

        return image, centroids 

def show_centroids(image, centroids, tensor=False):
    """Show image with centroids. 
        Arguments:
            -image: PIL Image 
            -centroids: np.array()
    """
    plt.figure()
    if tensor:
        image = image.permute(1, 2, 0)
    plt.imshow(image)
    plt.scatter(centroids[:, 1], centroids[:, 0], s=10, marker='.', c='r')
    plt.pause(0.001)  # pause a bit so that plots are updated
    plt.show()


