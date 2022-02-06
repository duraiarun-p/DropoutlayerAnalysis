#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  4 17:47:35 2022

@author: arun
"""
# %% Required Library Import Section
import os
import shutil
import torch
import numpy as np
from torchvision.io import read_image
from torch.utils.data import Dataset
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import time
import datetime

st_0 = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S') 
start_time_0=time.time()
print('Script started at')
print(st_0)
# %% Data Splitter function using ratio as the input argument


def split_dataset_into_2(path_to_dataset, train_ratio, test_ratio):
    """
    split the dataset in the given path into two subsets(test,train)
    :param path_to_dataset:
    :param train_ratio:
    :param test_ratio:
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
                              round(sub_dir_item_cnt[i] * (train_ratio + test_ratio))):
            if not os.path.exists(dir_valid_dst):
                os.makedirs(dir_valid_dst)

            source_file = os.path.join(sub_dir, items[item_idx])
            dst_file = os.path.join(dir_valid_dst, items[item_idx])
            shutil.copyfile(source_file, dst_file)
    return sub_dir_item_cnt

# %% Dataloader for EUROSat data


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
        image = image[None, :, :, :].float()
        label_index = self.img_labels_index[idx]
        label_vector = np.zeros_like(self.img_labels_index)
        label_vector[label_index] = 1
        label_vector = torch.from_numpy(label_vector)
        label_vector = label_vector.float()
        # if self.transform:
        #     image = self.transform(image)
        # if self.target_transform:
        #     label = self.target_transform(label)
        return image, label_vector

# %% Building Neural Network Models


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
    
class Net_drop_0dot5(nn.Module):
    def __init__(self, no_of_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, stride=1)
        self.conv2 = nn.Conv2d(16, 32, 3, stride=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)
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

class Net_drop_0dot7(nn.Module):
    def __init__(self, no_of_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, stride=1)
        self.conv2 = nn.Conv2d(16, 32, 3, stride=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.7)
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

class Net_drop_0dot9(nn.Module):
    def __init__(self, no_of_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, stride=1)
        self.conv2 = nn.Conv2d(16, 32, 3, stride=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.9)
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
    
class Net_drop_0dot99(nn.Module):
    def __init__(self, no_of_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, stride=1)
        self.conv2 = nn.Conv2d(16, 32, 3, stride=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.99)
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
    
class Net_drop_0dot999(nn.Module):
    def __init__(self, no_of_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, stride=1)
        self.conv2 = nn.Conv2d(16, 32, 3, stride=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.999)
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
# %% Class instantiation for Neural network building, training and performance evaluation


class EURONet():

    # Training loop
    def train(self):
        # Training Step
        def train_step():
            self.net.train(True)
            running_loss = 0.0
            batch_iteration_i_index = []
            batch_i_index = []
            for batch_iteration_i in range(self.train_batch_iteration):
                for batch_i in range(self.train_batch_size):

                    # basic training loop
                    inputs, labels = next(iter(train_dataset))
                    inputs, labels = inputs.to(device), labels.to(device)
                    self.optimizer.zero_grad()
                    outputs = self.net(inputs)
                    loss = self.criterion(outputs, labels)
                    loss.backward()
                    self.optimizer.step()

                    running_loss += loss.item()
                    batch_i_index.append(batch_i)
                batch_iteration_i_index.append(batch_iteration_i)
                return running_loss/(self.train_batch_iteration*self.train_batch_size)
        # Testing step

        def test_step():
            correct = 0
            # total = 0
            running_vloss = 0.0
            self.net.train(False)  # Stop training
            for batch_iteration_i in range(self.test_batch_iteration):
                for batch_i in range(self.test_batch_size):
                    vinputs, vlabels = next(iter(test_dataset))
                    vinputs, vlabels = vinputs.to(device), vlabels.to(device)
                    voutputs = self.net(vinputs)
                    vloss = self.criterion(voutputs, vlabels)
                    running_vloss += vloss.item()
                    _, predicted = torch.max(voutputs.data, 0)
                    _, target = torch.max(vlabels, 0)
                    correct += (predicted == target)
            avg_vloss = running_vloss / \
                (self.test_batch_iteration*self.test_batch_size)
            acc = (100*correct)/(self.test_batch_iteration*self.test_batch_size)
            return avg_vloss, acc

        # Class-wise accuracy check
        def class_wise_acc():
            total_class_bin = np.zeros_like(label_indices)
            correct_pred_class_bin = np.zeros_like(label_indices)
            with torch.no_grad():
                for batch_iteration_i in range(self.test_batch_iteration):
                    for batch_i in range(self.test_batch_size):
                        vinputs, vlabels = next(iter(test_dataset))
                        voutputs = self.net(vinputs)
                        _, predicted = torch.max(voutputs.data, 0)
                        _, target = torch.max(vlabels, 0)
                        total_class_bin[target] += 1
                        if target == predicted:
                            correct_pred_class_bin[target] += 1
            return correct_pred_class_bin, total_class_bin

        writer = SummaryWriter(self.log_name)
        train_len_list = []
        test_len_list = []
        total_data_len = 0
        train_data_len = 0
        test_data_len = 0
        for li in range(len(label_indices)):
            train_len_list.append(data_len_list[li]*0.8)
            test_len_list.append(data_len_list[li]*0.2)
            total_data_len = total_data_len+data_len_list[li]
            train_data_len = train_data_len+(data_len_list[li]*0.8)
            test_data_len = test_data_len+(data_len_list[li]*0.2)

        self.train_batch_iteration = round(
            train_data_len/self.train_batch_size)
        self.test_batch_iteration = round(test_data_len/self.test_batch_size)
        # Parameters for Training
        self.net = self.net.to(device)
        # criterion = self.criterion
        # optimizer = self.optimizer
        self.train_batch_iteration = 2
        # epochs=self.epochs
        self.test_batch_iteration = 2
        train_loss_epochs = np.zeros((self.epochs, 1))
        test_loss_epochs = np.zeros((self.epochs, 1))
        accuracy_epochs = np.zeros((self.epochs, 1))
        # Training for No. of epochs
        for epoch in range(self.epochs):
            train_loss = train_step()
            test_loss, accuracy = test_step()
            train_loss_epochs[epoch] = train_loss
            test_loss_epochs[epoch] = test_loss
            accuracy_epochs[epoch] = accuracy
            writer.add_scalars('Training vs. Validation Loss',
                               {'Epoch': epoch, 'Training': train_loss, 'Validation': test_loss})
        writer.flush()
        correct_pred_class_bin, total_class_bin = class_wise_acc()
        class_acc = (correct_pred_class_bin/total_class_bin)*100
        result_dic={"train_loss_epochs":train_loss_epochs, "test_loss_epochs":test_loss_epochs, 
                    "accuracy_epochs":accuracy_epochs, "correct_pred_class_bin":correct_pred_class_bin, 
                    "total_class_bin":total_class_bin, "class_acc":class_acc}
        print('Trained Successfully')
        return result_dic

    # Initialiation
    def __init__(self, net, optimizer, criterion, train_batch_size, test_batch_size, epochs, log_name):
        self.net = net
        self.epochs = epochs
        self.log_name = log_name
        self.optimizer = optimizer
        self.criterion = criterion
        self.test_batch_size = test_batch_size
        self.train_batch_size = train_batch_size


# %% Dataset preparation, Device selection and Summary Initialisation
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

datapath = os.path.join(os.getcwd(), '2750')  # Dataset path
label_names = os.listdir(datapath)
label_indices = np.arange(len(label_names))
no_of_classes = len(label_indices)

data_len_list = split_dataset_into_2(datapath, 0.8, 0.2)


datapath1 = os.path.join(os.getcwd(), 'train')
train_dataset = CustomImageDataset(label_indices, label_names, datapath1)
train_img0, train_label0 = next(iter(train_dataset))

datapath2 = os.path.join(os.getcwd(), 'test')
test_dataset = CustomImageDataset(label_indices, label_names, datapath2)
train_img1, train_label1 = next(iter(test_dataset))
img_shape = train_img0.shape  # Required for the model building
# %% All networks with dropouts
st_0 = datetime.datetime.fromtimestamp(
    time.time()).strftime('%Y-%m-%d %H:%M:%S')

train_batch_size = 1000
test_batch_size = 1000
epochs = 10
criterion = nn.MSELoss()

# Net without dropout
log_name1 = 'EuroNet_0_'+format(st_0)
net1 = Net_drop_0(no_of_classes)
optimizer1 = optim.Adam(net1.parameters(),lr=0.001,betas=(0.9, 0.999))
EuroNet_1 = EURONet(net1, optimizer1, criterion,
                    train_batch_size, test_batch_size, epochs, log_name1)
EuroNet_1_result_dic = EuroNet_1.train()

log_name2 = 'EuroNet_0dot5_'+format(st_0)
net2 = Net_drop_0dot5(no_of_classes)
optimizer2 = optim.Adam(net2.parameters(),lr=0.001,betas=(0.9, 0.999))
EuroNet_2 = EURONet(net2, optimizer2, criterion,
                    train_batch_size, test_batch_size, epochs, log_name2)
EuroNet_2_result_dic = EuroNet_2.train()

log_name3 = 'EuroNet_0dot7_'+format(st_0)
net3 = Net_drop_0dot7(no_of_classes)
optimizer3 = optim.Adam(net3.parameters(),lr=0.001,betas=(0.9, 0.999))
EuroNet_3 = EURONet(net3, optimizer3, criterion,
                    train_batch_size, test_batch_size, epochs, log_name3)
EuroNet_3_result_dic = EuroNet_3.train()

log_name4 = 'EuroNet_0dot9_'+format(st_0)
net4 = Net_drop_0dot9(no_of_classes)
optimizer4 = optim.Adam(net4.parameters(),lr=0.001,betas=(0.9, 0.999))
EuroNet_4 = EURONet(net4, optimizer4, criterion,
                    train_batch_size, test_batch_size, epochs, log_name4)
EuroNet_4_result_dic = EuroNet_4.train()

log_name5 = 'EuroNet_0dot99_'+format(st_0)
net5 = Net_drop_0dot99(no_of_classes)
optimizer5 = optim.Adam(net5.parameters(),lr=0.001,betas=(0.9, 0.999))
EuroNet_5 = EURONet(net5, optimizer5, criterion,
                    train_batch_size, test_batch_size, epochs, log_name5)
EuroNet_5_result_dic = EuroNet_5.train()

log_name6 = 'EuroNet_0dot999_'+format(st_0)
net6 = Net_drop_0dot999(no_of_classes)
optimizer6 = optim.Adam(net6.parameters(),lr=0.001,betas=(0.9, 0.999))
EuroNet_6 = EURONet(net6, optimizer5, criterion,
                    train_batch_size, test_batch_size, epochs, log_name6)
EuroNet_6_result_dic = EuroNet_6.train()
#%%
from matplotlib import pyplot as plt

fig1=plt.figure(1),
plt.plot(EuroNet_1_result_dic["accuracy_epochs"],label="dropout = 0")
plt.plot(EuroNet_2_result_dic["accuracy_epochs"],label="dropout = 0.5")
plt.plot(EuroNet_3_result_dic["accuracy_epochs"],label="dropout = 0.7")
plt.plot(EuroNet_4_result_dic["accuracy_epochs"],label="dropout = 0.9")
plt.plot(EuroNet_5_result_dic["accuracy_epochs"],label="dropout = 0.99")
plt.plot(EuroNet_6_result_dic["accuracy_epochs"],label="dropout = 0.999")
plt.xlabel('No. of epochs')
plt.ylabel('Accuracy')
plt.title('Accuracy')
plt.legend()

fig2=plt.figure(2),
plt.plot(EuroNet_1_result_dic["train_loss_epochs"],label="dropout = 0")
plt.plot(EuroNet_2_result_dic["train_loss_epochs"],label="dropout = 0.5")
plt.plot(EuroNet_3_result_dic["train_loss_epochs"],label="dropout = 0.7")
plt.plot(EuroNet_4_result_dic["train_loss_epochs"],label="dropout = 0.9")
plt.plot(EuroNet_5_result_dic["train_loss_epochs"],label="dropout = 0.99")
plt.plot(EuroNet_6_result_dic["train_loss_epochs"],label="dropout = 0.999")
plt.xlabel('No. of epochs')
plt.ylabel('Training loss')
plt.title('Training loss')
plt.legend()

fig3=plt.figure(3),
plt.plot(EuroNet_1_result_dic["test_loss_epochs"],label="dropout = 0")
plt.plot(EuroNet_2_result_dic["test_loss_epochs"],label="dropout = 0.5")
plt.plot(EuroNet_3_result_dic["test_loss_epochs"],label="dropout = 0.7")
plt.plot(EuroNet_4_result_dic["test_loss_epochs"],label="dropout = 0.9")
plt.plot(EuroNet_5_result_dic["test_loss_epochs"],label="dropout = 0.99")
plt.plot(EuroNet_6_result_dic["test_loss_epochs"],label="dropout = 0.999")
plt.xlabel('No. of epochs')
plt.ylabel('Test loss')
plt.title('Test loss')
plt.legend()
#%% Class wise accuracy
print('Class index is %s'%label_indices)
print('Class accuracy of NN with 0.0 drop',EuroNet_1_result_dic["class_acc"])
print('Class accuracy of NN with 0.5 drop',EuroNet_2_result_dic["class_acc"])
print('Class accuracy of NN with 0.7 drop',EuroNet_3_result_dic["class_acc"])
print('Class accuracy of NN with 0.9 drop',EuroNet_4_result_dic["class_acc"])
print('Class accuracy of NN with 0.99 drop',EuroNet_5_result_dic["class_acc"])
print('Class accuracy of NN with 0.999 drop',EuroNet_6_result_dic["class_acc"])
#%% Network classification accuracy
print('Net accuracy of NN with 0.0 drop',np.mean(EuroNet_1_result_dic["class_acc"]))
print('Net accuracy of NN with 0.5 drop',np.mean(EuroNet_2_result_dic["class_acc"]))
print('Net accuracy of NN with 0.7 drop',np.mean(EuroNet_3_result_dic["class_acc"]))
print('Net accuracy of NN with 0.9 drop',np.mean(EuroNet_4_result_dic["class_acc"]))
print('Net accuracy of NN with 0.99 drop',np.mean(EuroNet_5_result_dic["class_acc"]))
print('Net accuracy of NN with 0.999 drop',np.mean(EuroNet_6_result_dic["class_acc"]))
#%%
# import matplotlib
fig4=plt.figure(4),
sample_test_batch_size=6
for batch_i in range(sample_test_batch_size):
    vinputs, vlabels = next(iter(test_dataset))
    vinputs, vlabels = vinputs.to(device), vlabels.to(device)
    voutputs = net1(vinputs)
    _, predicted = torch.max(voutputs.data, 0)
    _, target = torch.max(vlabels, 0)
    vinputs_sq=torch.squeeze(vinputs)
    vinputs_sq=torch.permute(vinputs_sq, (1, 2, 0)).numpy()
    vinputs_sq=vinputs_sq.astype('uint8')
    # vinputs_sq=
    plt.subplot(2,3,batch_i+1),
    plt.imshow(vinputs_sq,cmap='gray')
    plt.show()
    titstr='Target:'+str(target.numpy())+' Predicted:'+str(predicted.numpy())
    plt.title(titstr)
    

#%%
print('Script started at')
print(st_0)
runtimeN0=(time.time()-start_time_0)/60
# runtimeN0=(time.time()-start_time_0)
print('Script Total Time = %s min'%(runtimeN0))
print('Script ended at')
st_0 = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
print(st_0)
