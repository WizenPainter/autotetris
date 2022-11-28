
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

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

# Loss function
class PadMSEloss(nn.MSELoss):
    def __init__(self, weight=None, size_average=None, reduce=None, reduction: str = 'mean'):
        super().__init__(size_average, reduce, reduction)

    def forward(self, input, target, ignore_index = 0):
        drop = target == ignore_index
        print(input.shape, target.shape)
        print(drop)
        input = input[~drop]
        target = target[~drop]
        print(input, input.shape, target, target.shape)

        return F.mse_loss(input, target, reduction=self.reduction)


# Model training
def train_model(network, criterion, optimizer, num_epochs, train_loader, valid_loader, device = 'cpu'):
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
            centroids = centroids.view(centroids.size(0),-1).to(device)
            # print(centroids.shape)

            #print(images.shape)
            #print(image)

            predictions = network(images)

            # clear all the gradients before calculating them
            optimizer.zero_grad()

            # find the loss for the current step
            loss_train_step = criterion(predictions, centroids)

            # calculate the gradients
            loss_train_step.backward()

            # update the parameters
            optimizer.step()

            loss_train += loss_train_step.item()

            running_loss = loss_train/step
            # loss_train_batch.append(running_loss)
            loss_train_full.append(running_loss)

            print_overwrite(step, len(train_loader), running_loss, 'train')
            step = step + 1

        step = 1
        network.eval()
        with torch.no_grad():

            for images, centroids in valid_loader:

                #images, centroids = next(iterator_valid)

                images = images.to(device)
                centroids = centroids.view(centroids.size(0),-1).to(device)

                predictions = network(images)

                # find the loss for the current step
                loss_valid_step = criterion(predictions, centroids)

                loss_valid += loss_valid_step.item()
                running_loss = loss_valid/step
                loss_val_full.append(running_loss)

                print_overwrite(step, len(valid_loader), running_loss, 'valid')
                step = step + 1

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
        predictions.append(prediction.view(-1,50,2))

        # Prepare data for plotting
        image = image.squeeze()
        image = image.permute(1, 2, 0)

        centroids = centroids.numpy()
        centroids = centroids[0]

        prediction = prediction.view(-1, 50, 2)
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
