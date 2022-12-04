
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from scipy.ndimage import zoom 

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
        resnet = models.resnet18(pretrained=False)
        # modules = list(resnet.children())[:-2]
        # self.model = nn.Sequential(*modules)
        self.model = resnet
        # self.gap = nn.AdaptiveAvgPool2d(1) #study how this works!! 
        self.fc1 = nn.Linear(1000, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.model(x)
        # final_conv_output = x
        x = self.fc1(x)
        x = self.fc2(x)
        # if heatmap:
        #     return final_conv_output, x
        return x

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
            
            #print(images.shape)
            #print(image)
            predictions = network(images)

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
            centroids = centroids.view(4,-1)
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
                centroids = centroids.view(centroids.size(0),-1).to(device)


                predictions = network(images)

                # find the loss for the current step
                try:
                    loss_valid_step = criterion(predictions, centroids)
                except:
                    print(predictions.shape)
                    print(centroids.shape)

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


def plot_activation_map_coordinates(model, test_loader, num_tests):

    iterator_c = iter(test_loader)
    image, centroids = next(iterator_c)

    weights = model.fc.weight #Get weights of the last layer 

    image = image.squeeze()
    image = image.permute(0, 2, 1)

    for i in range(1,num_tests + 1):
        conv_output, prediction = model(image, heatmap = True) #Get the output of the last conv layer and the network
        conv_output = np.squeeze(conv_output) 

        coordinate_weights = weights[i,:].detach() #The weights for one of the coordinates. NOTE: 0 could be changed

        mat_for_mult = zoom(conv_output.detach(), (32, 32, 1), order=1)
        final_output = np.dot(mat_for_mult.reshape((224*224, 512)), coordinate_weights).reshape(224,224) # dim: 224 x 224
        
        plt.imshow(image, alpha=0.5)
        plt.imshow(final_output, cmap='jet', alpha=0.5)

    prediction = prediction.view(-1, 50, 2)
    prediction = prediction.detach().numpy()
    prediction = prediction[0]
    plt.scatter(prediction[:, 1], prediction[:, 0], s=10, marker='.', c='r')

    plt.show()

def plot_activation_map_images(model, test_loader, num_tests, idx1, idx2):

    iterator_c = iter(test_loader)

    for step in range(1,num_tests + 1):
        image, centroids = next(iterator_c)
        conv_output, prediction = model(image, heatmap = True) #Get the output of the last conv layer and the network
        conv_output = np.squeeze(conv_output)

        prediction = prediction.view(-1, 37, 2)
        prediction = prediction.detach().numpy()
        prediction = prediction[0]

        weights = model.fc.weight #Get weights of the last layer 
        coordinate_x = weights[idx1,:].detach() #The weights for one of the coordinates. NOTE: 0 could be changed
        coordinate_y = weights[idx2,:].detach() #The weights for one of the coordinates. NOTE: 0 could be changed
        print(weights)
        # print(conv_output.shape)
        mat_for_mult = zoom(conv_output.detach(), (32, 32, 1), order=1)
        final_output_x = np.dot(mat_for_mult.reshape((224*224, 512)), coordinate_x).reshape(224,224) # dim: 224 x 224
        final_output_y = np.dot(mat_for_mult.reshape((224*224, 512)), coordinate_y).reshape(224,224) # dim: 224 x 224

        image = image.squeeze()
        image = image.permute(1, 2, 0)
        
        # print(image.shape)
        fig, ax = plt.subplots(1, 2)

        ax[0].imshow(image, alpha=0.5)
        ax[0].imshow(final_output_x, cmap='viridis', alpha=0.5)
        ax[0].set_title("X coordinate")
        ax[0].legend(loc="upper right")
        ax[0].axis("off")

        ax[1].imshow(image, alpha=0.5)
        ax[1].imshow(final_output_y, cmap='viridis', alpha=0.5)
        ax[1].set_title("Y coordinate")
        ax[1].legend(loc="upper right")
        ax[1].axis("off")

        ax[0].scatter(prediction[:, 1], prediction[:, 0], s=10, marker='.', c='r')
        ax[1].scatter(prediction[:, 1], prediction[:, 0], s=10, marker='.', c='r')

        plt.show()