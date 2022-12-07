
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from scipy.ndimage import zoom 
from kornia.geometry import subpix

import time
import matplotlib.pyplot as plt
import sys

# Model definition

class Resnet18(nn.Module):
    def __init__(self,num_classes=200):
        super().__init__()
        self.model_name='resnet18'
        self.model=models.resnet18()
        self.model.conv1=nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.model.fc=nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        x=self.model(x)
        return x
        
class Resnet50(nn.Module):
    def __init__(self,num_classes=200):
        super().__init__()
        self.model_name='resnet50'
        self.model=models.resnet50()
        self.model.conv1=nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.model.fc=nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        x=self.model(x)
        return x

class Resnet18_GAP(nn.Module): 
    def __init__(self,num_classes=200):
        super().__init__()
        self.model_name='resnet18'
        resnet = models.resnet18()
        modules = list(resnet.children())[:-2] #Removing the last two layers (up until before the Pooling)
        self.model = nn.Sequential(*modules)
        self.gap = nn.AdaptiveAvgPool2d(1) #Pooling down
        self.fc = nn.Linear(512, num_classes) #Last linear layer 

    def forward(self, x, heatmap = False):
        x = self.model(x)
        final_conv_output = x
        x = self.gap(x)
        x=self.fc(x.squeeze())
        if heatmap:
            return final_conv_output, x
        return x

class Resnet18_DSNT(nn.Module): 
    def __init__(self,num_classes=200):
        super().__init__()
        self.model_name='resnet18'
        resnet = models.resnet18()
        modules = list(resnet.children())[:-2] #Removing the last two layers (up until before the Pooling)
        self.model = nn.Sequential(*modules)
        self.hm_conv = nn.Conv2d(512, num_classes, kernel_size=1, bias=False)
        # self.gap = nn.AdaptiveAvgPool2d(1) #Pooling down
        # self.fc = nn.Linear(512, num_classes) #Last linear layer 

    def forward(self, x, heatmap = False):
        x = self.model(x)
        unnormalized_heatmaps = self.hm_conv(x)
        heatmaps = subpix.spatial_softmax2d(unnormalized_heatmaps)
        coords = subpix.spatial_expectation2d(heatmaps, normalized_coordinates=False)
        if heatmap:
            return  heatmaps, coords
        return coords

class SolarPanelDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.base_model = models.resnet50(pretrained=False)
        self.fc1 = nn.Linear(1000, 256)
        self.fc2 = nn.Linear(256, 224*224)

    def forward(self, x):
        x = self.base_model(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return x

class MaskRCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.detection.maskrcnn_resnet50_fpn(weights=True)

    def forward(self, x, y):
        self.model.eval()
        output = self.model(x, y)
        return output[0]['masks']


class DiceLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce_loss = nn.BCEWithLogitsLoss()

    def dice_loss(self, pred, target):
        smooth = 1.
        pred = pred.contiguous().view(-1)
        target = target.contiguous().view(-1)
        intersection = (pred * target).sum()
        return 1 - ((2. * intersection + smooth) / (pred.sum() + target.sum() + smooth))

    def forward(self, pred, target):
        return self.bce_loss(pred, target) + self.dice_loss(pred, target)




# Loss function
class PadMSEloss(nn.MSELoss):
    """MSE loss excluding padding values """
    def __init__(self, weight=None, size_average=None, reduce=None, reduction: str = 'mean'):
        super().__init__(size_average, reduce, reduction)

    def forward(self, input, target, ignore_index = 0):
        drop = (target == ignore_index) #create mask that drops padded values
        input = input[~drop]
        target = target[~drop]
        return F.mse_loss(input, target, reduction=self.reduction) 

class VarMSEloss(nn.MSELoss):
    """MSE loss taking into account the variance of the (x,y) coordinates.
            --> VarMSEloss(input, target) = MSE(input, target) + (Var(x_input) - Var(x_target))^2  + (Var(y_input) - Var(y_target))^2"""
    def __init__(self, weight=None, size_average=None, reduce=None, reduction: str = 'mean'):
        super().__init__(size_average, reduce, reduction)

    def forward(self, input, target, ignore_index = 0):
        drop = target == ignore_index
        input = input[~drop]
        target = target[~drop]

        reshaped_input = torch.transpose(input.view(-1,2), 0, 1)
        reshaped_target = torch.transpose(target.view(-1,2), 0, 1)

        var_x = torch.abs(torch.var(reshaped_input[0]) - torch.var(reshaped_target[0]))
        var_y = torch.abs(torch.var(reshaped_input[1]) - torch.var(reshaped_target[1]))
        
        return F.mse_loss(input, target, reduction=self.reduction) + var_x + var_y


class VarDiffloss(nn.MSELoss):
    """MSE loss taking into account the variance of the (x,y) coordinates.
            --> VarMSEloss(input, target) = MSE(input, target) + (Var(x_input) - Var(x_target))^2  + (Var(y_input) - Var(y_target))^2"""

    def __init__(self, weight=None, size_average=None, reduce=None, reduction: str = 'mean'):
        super().__init__(size_average, reduce, reduction)

    def forward(self, input, target, ignore_index=0):
        drop = target == ignore_index
        input = input[~drop]
        target = target[~drop]

        reshaped_input = torch.transpose(input.view(-1, 2), 0, 1)
        reshaped_target = torch.transpose(target.view(-1, 2), 0, 1)

        var_x = torch.var(reshaped_input[0] - reshaped_target[0])
        var_y = torch.var(reshaped_input[1] - reshaped_target[1])

        return F.mse_loss(input, target, reduction=self.reduction) + var_x + var_y


# Model training
def train_model(network, criterion, optimizer, num_epochs, train_loader, valid_loader, device = 'cpu', mask=False):
    loss_train_full = []
    loss_val_full = []
    start_time = time.time()

    for epoch in range(1,num_epochs+1):
        loss_train = 0
        loss_valid = 0
        running_loss = 0
        step = 1

        network.train()

        # initialize iterator
        #iterator_train = iter(train_loader)
        #iterator_valid = iter(valid_loader)

        for images, centroids in train_loader:

            #images, centroids = next(iterator_train)
            images = images.to(device)
            # centroids = centroids.view(centroids.size(0),-1).to(device)
            centroids = centroids.to(device)
            # print(centroids.shape)
            
            # print(images.shape)
            # print(centroids.shape)
            # predictions = network(images)
            predictions = network(images)
            # predictions = predictions[0]['masks']

            # if mask:
            #     masks = []
            #     for image in images:
            #         mask = torch.zeros_like(image[0])
            #         for coord in predictions:
            #             mask[int(coord[0])][int(coord[1])] = 1
            #         masks.append(mask)
            #     masks = torch.stack(masks)
            #     masks = masks.to(device)
            #     optimizer.zero_grad()
            #     masks = masks.view(4,1,224,224)
            #     loss_train_step = criterion(masks, centroids)

            # else:
            # clear all the gradients before calculating them
            optimizer.zero_grad()


            # find the loss for the current step
            # print(predictions.shape)
            # print(centroids.shape)
            # centroids = centroids.view(4,-1)
            # print(centroids.shape)
            loss_train_step = criterion(predictions, centroids)

            # calculate the gradients
            loss_train_step.backward()

            # update the parameters
            optimizer.step()

            loss_train += loss_train_step.item()

            running_loss = loss_train/step
            # loss_train_batch.append(running_loss)

            print_overwrite(step, len(train_loader), running_loss, 'train')
            step = step + 1
        loss_train_full.append(running_loss)
        step = 1
        network.eval()
        with torch.no_grad():

            for images, centroids in valid_loader:

                #images, centroids = next(iterator_valid)

                images = images.to(device)
                # centroids = centroids.view(centroids.size(0),-1).to(device)


                predictions = network(images)

                # find the loss for the current step
                # try:
                loss_valid_step = criterion(predictions, centroids)
                # except:
                #     print(predictions.shape)
                #     print(centroids.shape)

                loss_valid += loss_valid_step.item()
                running_loss = loss_valid/step
                # loss_valid_batch.append(running_loss)

                print_overwrite(step, len(valid_loader), running_loss, 'valid')
                step = step + 1
            loss_val_full.append(running_loss)

        loss_train /= len(train_loader)
        loss_valid /= len(valid_loader)
        # loss_train_full.append(loss_train)
        # loss_val_full.append(loss_valid)

        print('\n--------------------------------------------------')
        print('Epoch: {}  Train Loss: {:.4f}  Valid Loss: {:.4f}'.format(epoch, loss_train, loss_valid))
        print('--------------------------------------------------')

    print('Training Complete')
    print("Total Elapsed Time : {} s".format(time.time()-start_time))
    np.savetxt('train_loss', loss_train_full, delimiter=',')
    np.savetxt('val_loss', loss_val_full, delimiter=',')
    return network


# Help functions
def print_overwrite(step, total_step, loss, operation):
    sys.stdout.write('\r')
    if operation == 'train':
        sys.stdout.write("Train Steps: %d/%d  Loss: %.4f " % (step, total_step, loss))
    else:
        sys.stdout.write("Valid Steps: %d/%d  Loss: %.4f " % (step, total_step, loss))

    sys.stdout.flush()


def test_model(model, test_loader, num_tests):
    predictions = []
    iterator_c = iter(test_loader)
    for step in range(1,num_tests + 1):
        image, centroids = next(iterator_c)
        prediction = model(image)
        predictions.append(prediction.view(-1,37,2))

        # Prepare data for plotting
        image = image.squeeze()
        image = image.permute(1, 2, 0)

        centroids = centroids.numpy()
        centroids = centroids[0]

        prediction = prediction.view(-1, 37, 2)
        prediction = prediction.detach().numpy()
        prediction = prediction[0]

        fig, ax = plt.subplots(1, 2)
        ax[0].imshow(image)
        ax[0].scatter(centroids[:, 1], centroids[:, 0], s=10, marker='.', c='r')
        ax[0].set_title("Actual centroids")
        ax[0].axis("off")

        ax[1].imshow(image)
        ax[1].scatter(prediction[:, 1], prediction[:, 0], s=10, marker='.', c='r')
        ax[1].set_title("Predicted centroids")
        ax[1].axis("off")

        plt.show()

    return predictions

def test_model_heat(model, test_loader, num_tests):
    predictions = []
    iterator_c = iter(test_loader)
    for step in range(1,num_tests + 1):
        image, centroids = next(iterator_c)
        prediction = model(image)
        # print(prediction.shape)
        predictions.append(prediction.view(1,224,224))

        # Prepare data for plotting
        image = image.squeeze()
        image = image.permute(1, 2, 0)

        centroids = centroids.numpy()
        centroids = centroids[0]

        prediction = prediction.view(1,224,224)
        prediction = prediction.detach().numpy()
        prediction = prediction[0]

        fig, ax = plt.subplots(1, 2)
        ax[0].imshow(image)
        ax[0].scatter(centroids[:, 1], centroids[:, 0], s=10, marker='.', c='r')
        ax[0].set_title("Actual centroids")
        ax[0].axis("off")

        ax[1].imshow(image)
        plt.imshow(prediction, alpha=0.8)
        ax[1].set_title("Predicted centroids")
        ax[1].axis("off")

        plt.show()

def plot_CAM(model, test_loader, num_tests):
    """Plots the Class Activation Mappings for a model """
    iterator_c = iter(test_loader)

    for step in range(num_tests):
        image, centroid = next(iterator_c)
        conv_output, prediction = model(image, heatmap = True) #Get the output of the last conv layer and the network
        
        map = torch.mean(conv_output, dim = (0,1)) #Averaging over features 
        map = map.unsqueeze(dim=0)
        map = map.unsqueeze(dim=1)  #Expanidng channels 
        upscale = torch.nn.Upsample(scale_factor=32, mode='bicubic', recompute_scale_factor=True) #Upscaling
        map = upscale(map)
        map = map/torch.max(map) #Normalizing

        # Preprocessing data for graphing
        image_plot = image.squeeze()
        image_plot = image_plot.permute(1, 2, 0)
        # m2 = map.transpose(0,1)

        prediction = prediction.view(-1, 37, 2)
        prediction = prediction.detach().numpy()
        prediction = prediction[0]

        centroid_plot = centroid.squeeze()
        
        #Plotting 
        fig, ax = plt.subplots(1, 2, figsize=(8, 8))
        cbar = fig.colorbar(plt.cm.ScalarMappable(cmap="jet"), location = "bottom")
        ax[0].imshow(image_plot)
        ax[0].imshow(map[0][0].detach(), alpha=0.3, cmap ='jet')

        ax[0].scatter(prediction[:, 1], prediction[:, 0], s=10, marker='.', c='r')
        ax[0].set_title("Model Prediction")

        ax[1].imshow(image_plot)
        ax[1].scatter(centroid_plot[:, 1], centroid_plot[:, 0], s=10, marker='.', c='b')
        ax[1].set_title("Reality")

        plt.show()

def plot_losses(train_loss, val_loss):
    """Plot the training and validation loss.
        - train_loss: np.array
        - val_loss: np.array
    """
    fig, ax = plt.subplots(2, 1, figsize=(8, 8))

    x1 = np.arange(len(train_loss))
    x2 = np.arange(len(val_loss))

    ax[0].plot(x1, train_loss)
    ax[0].set_title("Training loss")

    ax[1].plot(x2, val_loss)
    ax[1].set_title("Validation loss")

    plt.show()