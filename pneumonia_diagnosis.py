#!/usr/bin/env python
# coding: utf-8

# # Pneumonia Diagnosis

# The task is to predict if a person has pneumonia or not using Chest X-Ray.
# 
# We will train a Convolutional Neural Network (CNN) that is able to detect whether a patient has pneumonia, both bacterial and viral, based on an X-ray image of their chest. We need to classify a patient as either having pneumonia or not having pneumonia. This is a binary classification problem.

# **Credits**: Kaggle (https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia)

# First, we will create a CNN from scratch and check the test accuracy. And then, we will use transfer learning (using a DenseNet-169 pre-trained model) to create a CNN that will greatly improve the test accuracy.

# In[1]:


get_ipython().system(' wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate \'https://docs.google.com/uc?export=download&id=1li6ctqAvGFgIGMSt-mYrLoM_tbYkzqdO\' -O- | sed -rn \'s/.*confirm=([0-9A-Za-z_]+).*/\\1\\n/p\')&id=1li6ctqAvGFgIGMSt-mYrLoM_tbYkzqdO" -O chest_xray.zip && rm -rf /tmp/cookies.txt')


# In[2]:


get_ipython().system('unzip -qq chest_xray.zip')


# In[3]:


# importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from collections import Counter
from datetime import datetime

import torch
import torch.nn.functional as F

from torchvision import transforms, datasets, models
from torch import nn, optim


# In[4]:


# specify the data directory path
data_dir = './chest_xray'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/val'
test_dir = data_dir + '/test'


# In[5]:


# check if CUDA support is available
use_cuda = torch.cuda.is_available()
print('Cuda support available? - {}'.format(use_cuda))


# In[6]:


# normalization supported by transfer learning models
normalize = transforms.Normalize([0.485, 0.456, 0.406], 
                                 [0.229, 0.224, 0.225])

# transform the data
train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize
])
valid_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize
])
test_transforms = transforms.Compose([
    transforms.Resize(size=(224, 224)),
    transforms.ToTensor(),
    normalize
])


# I have applied the RandomResizedCrop and RandomHorizontalFlip to the training data. This will allow me to have more images using image augmentation techniques. It will generate more resized and flipped images. It will improve the performance of model and also helps to prevent overfitting of the data. For validation data, I have only applied the Resize and center crop transformations. And, for test data, I have only applied image resize.

# In[7]:


# specify the image folders
train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
valid_data = datasets.ImageFolder(valid_dir, transform=valid_transforms)
test_data = datasets.ImageFolder(test_dir, transform=test_transforms)


# In[8]:


batch_size = 32  # samples per batch
num_workers = 0  # number of subprocesses

# data loaders
trainloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, 
                                          shuffle=True, num_workers=num_workers)
validloader = torch.utils.data.DataLoader(valid_data, batch_size=batch_size, 
                                          shuffle=True, num_workers=num_workers)
testloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, 
                                         num_workers=num_workers)


# In[48]:


# select a batch from training data
images, labels = next(iter(trainloader))


# In[10]:


# shape of an image
images[0].shape  # rgb image of 244 x 244


# In[11]:


# output classes
print(train_data.classes)
print(train_data.class_to_idx)


# We will now have a look at the distribution of samples in the training, validation and testing dataset.

# In[12]:


# distribution of train dataset
cnt = Counter()
for target in train_data.targets:
    cnt[target] += 1

normal_count = cnt[0]
pneumonia_count = cnt[1]

sns.barplot(x=['Pneumonia Cases', 'Normal Cases'], 
            y=[pneumonia_count, normal_count], palette='magma')
plt.title('Train Dataset Label Count')
plt.show()
pneumonia_count, normal_count


# In[13]:


# distribution of validation dataset
cnt = Counter()
for target in valid_data.targets:
    cnt[target] += 1

normal_count = cnt[0]
pneumonia_count = cnt[1]

sns.barplot(x=['Pneumonia Cases', 'Normal Cases'], 
            y=[pneumonia_count, normal_count], palette='magma')
plt.title('Validation Dataset Label Count')
plt.show()
pneumonia_count, normal_count


# In[14]:


# distribution of test dataset
cnt = Counter()
for target in test_data.targets:
    cnt[target] += 1

normal_count = cnt[0]
pneumonia_count = cnt[1]

sns.barplot(x=['Pneumonia Cases', 'Normal Cases'], 
            y=[pneumonia_count, normal_count], palette='magma')
plt.title('Test Dataset Label Count')
plt.show()
pneumonia_count, normal_count


# We will have a look at the normal and pneumonia images of chest x-rays.

# In[51]:


num_classes = 2 # total classes of diagnosis (Normal, Pneumonia)
classes = ['NORMAL', 'PNEUMONIA']


# In[16]:


# un-normalize and display an image
def imshow(inp):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)


# In[53]:


# plot the images in the batch, along with predicted and true labels
fig = plt.figure(figsize=(25, 8))
for idx in np.arange(10):
    ax = fig.add_subplot(2, 10/2, idx+1, xticks=[], yticks=[])
    imshow(images.cpu()[idx])
    ax.set_title(classes[labels[idx]])


# Let's create a CNN from scratch and check the test accuracy. Then we will try to improve the accuracy using transfer learning.

# In[18]:


# CNN architecture
class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        ## cnn layers
        self.conv1 = nn.Conv2d(3, 32, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)

        # max-pool
        self.pool = nn.MaxPool2d(2, 2)
        
        # fully-connected
        self.fc1 = nn.Linear(7 * 7 * 128, 512)
        self.fc2 = nn.Linear(512, 512) 
        self.fc3 = nn.Linear(512, num_classes) 
        
        # drop-out
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, x):

        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        
        # flatten the images with batch
        x = x.view(-1, 7 * 7 * 128)
        
        x = self.dropout(x)
        x = F.relu(self.fc1(x))

        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        
        x = self.dropout(x)
        x = self.fc3(x)
        return x


# In[19]:


# instantiate the CNN
model_init = Net()
print(model_init)

# move tensors to GPU if CUDA is available
if use_cuda:
    model_init.cuda()


# The first convolution layer will have a kernel size of 3 and stride 2, this will decrease the input image size by half. The second convolution layer will also have a kernel size of 3 and stride 2, which will decrease the input image size by half. The third convolution layer will have a kernel size of 3.
# 
# I have applied the max-pooling of stride 2 after each convolution layer to reduce the image size by half. I have also applied Relu activation for each of the convolution layers.
# 
# Then, I have flattened the inputs and applied a dropout layer with probability as 0.3. Three fully connected layers are applied with Relu activation and dropout 0.3 to produce the final output that will predict the classes of a chest x-ray.

# In[20]:


# define loss function criteria and optimizer 
criterion_init = nn.CrossEntropyLoss()
optimizer_init = optim.Adam(model_init.parameters(), lr=0.03)


# Let's define a function to train the model and save the final model parameters as 'model_init.pt'

# In[21]:


def train(n_epochs, train_loader, valid_loader, model, optimizer, criterion, 
          use_cuda, save_path):

    valid_loss_min = np.Inf  # initialize inital loss to infinity
    train_loss_list = []
    valid_loss_list = []

    for epoch in range(1, n_epochs+1):
        epoch_start = datetime.utcnow()
        
        # initialize variables to monitor training and validation loss
        train_loss = 0.0
        valid_loss = 0.0

        # train the model
        model.train()
        for data, target in train_loader:

            # move to GPU
            if use_cuda:
                data, target = data.cuda(), target.cuda()

            # initialize weights to zero
            optimizer.zero_grad()

            # predict the output
            output = model(data)

            # calculate loss
            loss = criterion(output, target)

            # backpropogation
            loss.backward()

            # update gradients
            optimizer.step()

            train_loss += loss.item() * data.size(0)

        # validate the model
        model.eval()

        for data, target in valid_loader:

            if use_cuda:
                data, target = data.cuda(), target.cuda()

            output = model(data)
            loss = criterion(output, target)
            valid_loss = loss.item() * data.size(0)

        # calculate average losses
        train_loss = train_loss/len(train_loader.sampler)
        valid_loss = valid_loss/len(valid_loader.sampler)
        
        train_loss_list.append(train_loss)
        valid_loss_list.append(valid_loss)
        
        print('Epoch training time: {}'.format(datetime.utcnow() - epoch_start))
        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
            epoch, train_loss, valid_loss))

        # save the model if validation loss has decreased
        if valid_loss < valid_loss_min:
            torch.save(model.state_dict(), save_path)
            print('Model saved!\t\tValidation loss decreased ({:.6f} -> {:.6f})'.format(
                valid_loss_min, valid_loss))
            valid_loss_min = valid_loss

    # plot the training and validation loss
    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(range(len(train_loss_list)), train_loss_list, 
                label='Training Loss')
    plt.plot(range(len(valid_loss_list)), valid_loss_list, 
                label='Validation Loss')
    plt.legend(loc='upper right')
    plt.xlabel('Number of epochs')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Loss')

    # return the trained model
    return model


# In[22]:


# train the model
start = datetime.utcnow()
model_init = train(10, trainloader, validloader, model_init, optimizer_init,
                   criterion_init, use_cuda, 'model_init.pt')
print("model_init training time: {}".format(datetime.utcnow() - start))


# In[23]:


# load the model that got the best validation accuracy
model_init.load_state_dict(torch.load('model_init.pt'))


# In[24]:


def test(test_loader, model, criterion, use_cuda):
    test_loss = 0.
    correct = 0.
    total = 0.

    for data, target in test_loader:
        # move to GPU
        if use_cuda:
            data, target = data.cuda(), target.cuda()
        # predict output
        output = model(data)
        # calculate the loss
        loss = criterion(output, target)
        # update average test loss
        test_loss += loss.item() * data.size(0)
        # convert output probabilities to predicted class
        pred = output.data.max(1, keepdim=True)[1]
        # compare predictions to true label
        correct += np.sum(np.squeeze(pred.eq(target.data.view_as(pred))).cpu().numpy())
        total += data.size(0)

    print('Test Loss: {:.6f}\n'.format(test_loss/len(testloader.sampler)))
    print('\nTest Accuracy: %2d%% (%2d/%2d)' %
          (100. * correct / total, correct, total))


# In[25]:


# test the model
test(testloader, model_init, criterion_init, use_cuda)


# In[26]:


# visualize the confusion matrix
def plot_confusion_matrix(C):
    plt.figure(figsize=(20, 4))    
    labels = [0, 1]
    cmap=sns.light_palette("blue")
    plt.subplot(1, 3, 1)
    sns.heatmap(C, annot=True, cmap=cmap, fmt=".0f", 
                xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted Class')
    plt.ylabel('Original Class')
    plt.title("Confusion matrix")


# In[58]:


# generate confustion matrix
confusion_matrix = torch.zeros(num_classes, num_classes)
with torch.no_grad():
    for i, (inputs, labels) in enumerate(testloader):
        if use_cuda:
            inputs = inputs.cuda()
            labels = labels.cuda()
        outputs = model_init(inputs)
        _, preds = torch.max(outputs, 1)
        for t, p in zip(labels.view(-1), preds.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1

print(confusion_matrix)


# In[59]:


# get the per-class accuracy
print(confusion_matrix.diag()/confusion_matrix.sum(1))


# In[60]:


# plot the confustion matrix
plot_confusion_matrix(confusion_matrix)


# Now, we will use transfer learning to create a CNN that can diagnose pneumonia from images.
# 
# We will use DenseNet-169 model as it has good performance on Image classification. The main idea of this model is called "identity shortcut connection" that skips one or more layers. This allows us to prevent overfitting while training. I have eventually added a final fully connected layer that will output the probabilities of 2 classes of normal or pneumonia.

# In[30]:


# download the pretraibed DenseNet-169 model
model = models.densenet169(pretrained=True)


# In[31]:


# freeze the model parameters
for param in model.parameters():
    param.requires_grad = False


# In[32]:


# check the number of input and output features
model.classifier


# We will keep the number of input features same, however we will change the number of output features to 2 as we want to predict only two classes i.e. Normal and Pneumonia.

# In[33]:


# update the out_features for model
model.classifier = nn.Linear(model.classifier.in_features, num_classes)


# In[34]:


fc_parameters = model.classifier.parameters()


# In[35]:


for param in fc_parameters:
    param.requires_grad = True


# In[36]:


# move model to gpu
if use_cuda:
    model = model.cuda()


# In[37]:


# DenseNet-169 model architecture
model


# In[38]:


# define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)


# We will use the same function to train the model and save the final model parameters as 'model.pt'

# In[39]:


# train the model
start = datetime.utcnow()
model = train(30, trainloader, validloader, model, optimizer, 
                   criterion, use_cuda, 'model.pt')
print("model training time: {}".format(datetime.utcnow() - start))


# In[40]:


# load the model that got the best validation accuracy
model.load_state_dict(torch.load('model.pt'))


# In[41]:


# test the model
test(testloader, model, criterion, use_cuda)


# Let's try to visualize the predict the final output of few a few xray images.

# In[42]:


dataiter = iter(testloader)
images, labels = dataiter.next()
images.numpy()

if use_cuda:
    images = images.cuda()

# get sample outputs
output = model(images)
# convert output probabilities to predicted class
_, preds_tensor = torch.max(output, 1)

if use_cuda:
    preds = np.squeeze(preds_tensor.cpu().numpy())
else:
    preds = np.squeeze(preds_tensor.numpy())

# plot the images in the batch, along with predicted and true labels
fig = plt.figure(figsize=(25, 8))
for idx in np.arange(10):
    ax = fig.add_subplot(2, 10/2, idx+1, xticks=[], yticks=[])
    imshow(images.cpu()[idx])
    ax.set_title("{} ({})".format(classes[preds[idx]], classes[labels[idx]]), 
                 color=("green" if preds[idx] == labels[idx].item() else "red"))


# In[61]:


# generate confustion matrix
confusion_matrix = torch.zeros(num_classes, num_classes)
with torch.no_grad():
    for i, (inputs, labels) in enumerate(testloader):
        if use_cuda:
            inputs = inputs.cuda()
            labels = labels.cuda()
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        for t, p in zip(labels.view(-1), preds.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1

print(confusion_matrix)


# In[62]:


# get the per-class accuracy
print(confusion_matrix.diag()/confusion_matrix.sum(1))


# In[63]:


# plot the confusion matrix
plot_confusion_matrix(confusion_matrix)


# In[43]:




