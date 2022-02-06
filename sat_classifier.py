#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  3 11:24:00 2022

@author: arun
"""

import os
import shutil
import torch
import torchvision
import numpy as np
import PIL
import scipy
from torchvision.io import read_image
from torch.utils.data import Dataset
# from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor

import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

# from torchsummary import summary # Only for Precision remove it for Colab
#%% Splitting Dataset into Training set and test Set

def split_dataset_into_3(path_to_dataset, train_ratio, valid_ratio):
    """
    split the dataset in the given path into three subsets(test,validation,train)
    :param path_to_dataset:
    :param train_ratio:
    :param valid_ratio:
    :return:
    """
    _, sub_dirs, _ = next(iter(os.walk(path_to_dataset))
                          )  # retrieve name of subdirectories
    # list for counting items in each sub directory(class)
    sub_dir_item_cnt = [0 for i in range(len(sub_dirs))]

    # directories where the splitted dataset will lie
    dir_train = os.path.join(os.path.dirname(path_to_dataset), 'train')
    dir_valid = os.path.join(os.path.dirname(path_to_dataset), 'test')
    dir_test = os.path.join(os.path.dirname(path_to_dataset), 'test')

    for i, sub_dir in enumerate(sub_dirs):

        # directory for destination of train dataset
        dir_train_dst = os.path.join(dir_train, sub_dir)
        # directory for destination of validation dataset
        dir_valid_dst = os.path.join(dir_valid, sub_dir)
        # directory for destination of test dataset
        dir_test_dst = os.path.join(dir_test, sub_dir)

        # variables to save the sub directory name(class name) and to count the images of each sub directory(class)
        class_name = sub_dir
        sub_dir = os.path.join(path_to_dataset, sub_dir)
        sub_dir_item_cnt[i] = len(os.listdir(sub_dir))

        items = os.listdir(sub_dir)

        # transfer data to trainset
        for item_idx in range(round(sub_dir_item_cnt[i] * train_ratio)):
            if not os.path.exists(dir_train_dst):
                os.makedirs(dir_train_dst)

            source_file = os.path.join(sub_dir, items[item_idx])
            dst_file = os.path.join(dir_train_dst, items[item_idx])
            shutil.copyfile(source_file, dst_file)

        # transfer data to validation
        for item_idx in range(round(sub_dir_item_cnt[i] * train_ratio) + 1,
                              round(sub_dir_item_cnt[i] * (train_ratio + valid_ratio))):
            if not os.path.exists(dir_valid_dst):
                os.makedirs(dir_valid_dst)

            source_file = os.path.join(sub_dir, items[item_idx])
            dst_file = os.path.join(dir_valid_dst, items[item_idx])
            shutil.copyfile(source_file, dst_file)

        # # transfer data to testset
        for item_idx in range(round(sub_dir_item_cnt[i] * (train_ratio + valid_ratio)) + 1, sub_dir_item_cnt[i]):
            if not os.path.exists(dir_test_dst):
                os.makedirs(dir_test_dst)

            source_file = os.path.join(sub_dir, items[item_idx])
            dst_file = os.path.join(dir_test_dst, items[item_idx])
            shutil.copyfile(source_file, dst_file)

    return


def split_dataset_into_2(path_to_dataset, train_ratio, valid_ratio):
    """
    split the dataset in the given path into three subsets(test,validation,train)
    :param path_to_dataset:
    :param train_ratio:
    :param valid_ratio:
    :return:
    """
    # retrieve name of subdirectories
    _, sub_dirs, _ = next(iter(os.walk(path_to_dataset)))
    # list for counting items in each sub directory(class)
    sub_dir_item_cnt = [0 for i in range(len(sub_dirs))]

    # directories where the splitted dataset will lie
    dir_train = os.path.join(os.path.dirname(path_to_dataset), 'train')
    dir_valid = os.path.join(os.path.dirname(path_to_dataset), 'test')

    for i, sub_dir in enumerate(sub_dirs):

        # directory for destination of train dataset
        dir_train_dst = os.path.join(dir_train, sub_dir)
        # directory for destination of validation dataset
        dir_valid_dst = os.path.join(dir_valid, sub_dir)

        # variables to save the sub directory name(class name) and
        # to count the images of each sub directory(class)
        sub_dir = os.path.join(path_to_dataset, sub_dir)
        sub_dir_item_cnt[i] = len(os.listdir(sub_dir))

        items = os.listdir(sub_dir)

        # transfer data to trainset
        for item_idx in range(round(sub_dir_item_cnt[i] * train_ratio)):
            if not os.path.exists(dir_train_dst):
                os.makedirs(dir_train_dst)

            source_file = os.path.join(sub_dir, items[item_idx])
            dst_file = os.path.join(dir_train_dst, items[item_idx])
            shutil.copyfile(source_file, dst_file)

        # transfer data to validation
        for item_idx in range(round(sub_dir_item_cnt[i] * train_ratio) + 1,
                              round(sub_dir_item_cnt[i] * (train_ratio + valid_ratio))):
            if not os.path.exists(dir_valid_dst):
                os.makedirs(dir_valid_dst)

            source_file = os.path.join(sub_dir, items[item_idx])
            dst_file = os.path.join(dir_valid_dst, items[item_idx])
            shutil.copyfile(source_file, dst_file)
    return sub_dir_item_cnt

#%% Data loader for Pytorch


class CustomImageDataset(Dataset):
    def __init__(self, label_indices, label_names, img_dir, transform=None,
                 target_transform=None):
        self.img_labels = label_names
        self.img_labels_index = label_indices
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        idx = np.random.choice(len(self.img_labels_index))
        img_folder_path = os.path.join(self.img_dir, self.img_labels[idx])
        img_files = os.listdir(img_folder_path)
        img_file_index = np.random.choice(len(img_files))
        img_path = img_files[img_file_index]
        image = read_image(os.path.join(img_folder_path, img_path))
        # Use this input syntax to avoid dtype runtime and dimension error
        image = image[None,:,:,:].float()
        label_index = self.img_labels_index[idx]
        label_vector = np.zeros_like(self.img_labels_index)
        label_vector[label_index]=1
        label_vector=torch.from_numpy(label_vector)
        label_vector=label_vector.float()
        # if self.transform:
        #     image = self.transform(image)
        # if self.target_transform:
        #     label = self.target_transform(label)
        return image, label_vector
#%% Dataset preparation, Device selection and Summary Initialisation

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

datapath = os.path.join(os.getcwd(), '2750')  # Dataset path
label_names = os.listdir(datapath)
label_indices = np.arange(len(label_names))
no_of_classes = len(label_indices)

data_len_list=split_dataset_into_2(datapath, 0.8, 0.2)


datapath1 = os.path.join(os.getcwd(), 'train')
train_dataset = CustomImageDataset(label_indices, label_names, datapath1)
train_img0, train_label0 = next(iter(train_dataset))

datapath2 = os.path.join(os.getcwd(), 'test')
test_dataset = CustomImageDataset(label_indices, label_names, datapath2)
train_img1, train_label1 = next(iter(test_dataset))
img_shape = train_img0.shape  # Required for the model building


#%% Building Neural Network Model

class Net_drop_0(nn.Module):
    def __init__(self, no_of_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, stride=1)
        self.conv2 = nn.Conv2d(16, 32, 3, stride=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.0)
        self.no_classes = no_of_classes
        self.op_fcl = nn.Linear(128, self.no_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))  # 1st Conv + Relu
        x = self.pool(F.relu(self.conv2(x)))  # 2nd Conv + Relu + Maxpool
        x = torch.flatten(x)
        fcl_shape = x.shape
        self.fc1 = nn.Linear(fcl_shape[0], 128)  # 1st Fully Conn. Layer
        x = F.relu(self.fc1(x))
        x = self.dropout(x)  # Apply Dropout
        x = self.op_fcl(x)  # Classification layer
        return x

#%% Training and testing step function definition
def train_step(net,optimizer,criterion,train_batch_iteration,train_batch_size):
    net.train(True)
    running_loss = 0.0
    batch_iteration_i_index=[]
    batch_i_index=[]
    for batch_iteration_i in range(train_batch_iteration):
        for batch_i in range(train_batch_size):
            
            # basic training loop
            inputs, labels = next(iter(train_dataset))
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    
            running_loss += loss.item()
            batch_i_index.append(batch_i)
        batch_iteration_i_index.append(batch_iteration_i)
        return running_loss/(train_batch_iteration*train_batch_size)
    
def test_step(net,criterion,test_batch_iteration,test_batch_size):
    correct = 0
    # total = 0
    running_vloss = 0.0
    net.train(False) # Stop training
    for batch_iteration_i in range(test_batch_iteration):
        for batch_i in range(test_batch_size):
            vinputs, vlabels = next(iter(test_dataset))
            vinputs, vlabels = vinputs.to(device), vlabels.to(device)
            voutputs = net(vinputs)
            vloss = criterion(voutputs, vlabels)
            running_vloss += vloss.item()
            _, predicted = torch.max(voutputs.data, 0)
            _, target = torch.max(vlabels, 0)
            correct += (predicted == target)           
    return running_vloss/(test_batch_iteration*test_batch_size),(100*correct)/(test_batch_iteration*test_batch_size)
#%%


#%% Batch preparation, Training and Performance logging
writer = SummaryWriter('zonda_experiment_1')
train_len_list = []
test_len_list = []
total_data_len = 0
train_data_len = 0
test_data_len = 0
for li in range(len(label_indices)):
    train_len_list.append(data_len_list[li]*0.8)
    test_len_list.append(data_len_list[li]*0.2)
    total_data_len=total_data_len+data_len_list[li]
    train_data_len=train_data_len+(data_len_list[li]*0.8)
    test_data_len=test_data_len+(data_len_list[li]*0.2)

train_batch_size=1000
train_batch_iteration=round(train_data_len/train_batch_size)
test_batch_size=1000
test_batch_iteration=round(test_data_len/test_batch_size)
# Parameters for Training
# criterion = nn.CrossEntropyLoss()
net = Net_drop_0(no_of_classes)
net = net.to(device)
criterion = nn.MSELoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
train_batch_iteration=1
epochs=2
test_batch_iteration=1
train_loss_epochs=np.zeros((epochs,1))
test_loss_epochs=np.zeros((epochs,1))
accuracy_epochs=np.zeros((epochs,1))
#Training for No. of epochs
for epoch in range(epochs):
    train_loss=train_step(net,optimizer,criterion,train_batch_iteration,train_batch_size)
    test_loss,accuracy=test_step(net,criterion,test_batch_iteration,test_batch_size)
    train_loss_epochs[epoch]=train_loss
    test_loss_epochs[epoch]=test_loss
    accuracy_epochs[epoch]=accuracy
    writer.add_scalars('Training vs. Validation Loss',
                            {'Epoch' : epoch, 'Training' : train_loss, 'Validation' : test_loss })
writer.flush()
#%% Classwise accuracy
def class_wise_acc(net):   
    total_class_bin = np.zeros_like(label_indices)
    correct_pred_class_bin = np.zeros_like(label_indices)
    with torch.no_grad():
        for batch_iteration_i in range(test_batch_iteration):
            for batch_i in range(test_batch_size):
                vinputs, vlabels = next(iter(test_dataset))
                voutputs = net(vinputs)
                _, predicted = torch.max(voutputs.data, 0)
                _, target = torch.max(vlabels, 0)
                total_class_bin[target]+=1            
                if target == predicted:
                    correct_pred_class_bin[target] += 1
    return correct_pred_class_bin,total_class_bin

correct_pred_class_bin,total_class_bin=class_wise_acc(net)
class_acc=(correct_pred_class_bin/total_class_bin)*100