####################################Import####################################
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.autograd import Variable
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.autograd import Variable
import torch.utils.data as Data
from PIL import Image 
import pandas as pd
from sklearn.metrics import confusion_matrix
#from sklearn.preprocessing import scale 

#####################################Training settings#####################################
parser = argparse.ArgumentParser(description='ULI dataset')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 164)')
parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                    help='learning rate (default: 1e-2)')
parser.add_argument('--momentum', type=float, default=0.1, metavar='M',
                    help='SGD momentum (default: 0.1)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
"""    
def read_data():
    training_data = pd.read_csv("trainingData.csv",header = 0)
    train_x = scale(np.asarray(training_data.loc[:,0:520]))
    train_y = np.asarray(training_data["BUILDINGID"].map(str) + training_data["FLOOR"].map(str))
    train_y = np.asarray(pd.get_dummies(train_y))
 
    test_dataset = pd.read_csv("validationData.csv",header = 0)
    test_x = scale(np.asarray(test_dataset.loc[:,0:520]))
    test_y = np.asarray(test_dataset["BUILDINGID"].map(str) + test_dataset["FLOOR"].map(str))
    test_y = np.asarray(pd.get_dummies(test_y))
    return train_x,train_y,test_x,test_y
    
train_data,train_label,test_data,test_label=read_data()
#print(train_data)
"""
def read_and_preprocess_data(file_dir):
    data = pd.read_csv(file_dir)
    sensor_columns = get_all_sensor_column_names()

    data.loc[:, sensor_columns] = \
        data.loc[:, sensor_columns].applymap(transform_signal_strength)
    print(data.head())

    # Categorize each row based on Building ID and Floor?
    data['position'] = data.apply(
        lambda row : str(int(row['BUILDINGID'])) + "_" + str(int(row['FLOOR'])), axis='columns')
    data['position'] = data['position'].astype('category')
    return data
training_data = read_and_preprocess_data('./trainingData.csv')

