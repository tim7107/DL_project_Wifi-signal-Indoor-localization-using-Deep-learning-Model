import torch
import matplotlib.pyplot as plt
import numpy as np
import sklearn
import lightgbm as lgb
from torch.utils import data
import gan
device = torch.device("cuda:0" )
class UJIDataset(torch.utils.data.Dataset):
    def __init__(self, root, train = True, transform = None, target_transform = None, download = False , if_gan = True ):
        self.root = root
        dir_path = self.root
        dataset_training_file = dir_path + '/trainingData.csv'
        dataset_validation_file = dir_path + '/validationData.csv'
        # Load independent variables (WAPs values)
        if train:
            dataset_file = dataset_training_file
        else:
            dataset_file = dataset_validation_file
        file = open(dataset_file, 'r')
        
        # Load labels
        label = file.readline()

        label = label.split(',')

        # Load independent variables
        file_load = np.loadtxt(file, delimiter = ',', skiprows = 0)


        # RSSI values
        self.x = file_load[:, 0 : 520]

        # Load dependent variables
        self.y = file_load[:, 520 : 524]
        # Divide labels into x and y
        self.x_label = label[0 : 520]
        self.x_label = np.concatenate([self.x_label, label[524: 529]])
        self.y_label = label[520 : 524]
        # Regularization of independent variables
        self.x[self.x == 100] = -104    # WAP not detected
        self.x = self.x + 104             # Convert into positive values
        self.x = self.x / 104             # Regularize into scale between 0 and 1
        # Building ID, Space ID, Relative Position, User ID, Phone ID and Timestamp respectively

        file.close()
        # Reduce the number of dependent variables by combining building number and floor into one variable: area
        # 5*buliding + floor 
        area = self.y[:, 3] * 4 + self.y[:, 2]
        #print(self.y[0].size)
        #print(area[0].size)
        #self.y = np.column_stack((self.y, area))
        self.y = area
        if if_gan==1 :
            number_generator = 10000
            generator_data,fixed_label = gan.generate(number_generator) 
            generator_data = generator_data.cpu()
            generator_data = generator_data.detach().numpy()
            fixed_label = fixed_label.cpu()
            fixed_label = fixed_label.detach().numpy()            
        #---------------------concatenate gan and label----------------------#
            fixed_label = fixed_label.flatten()
            self.y = np.concatenate([self.y,fixed_label]) 
            self.x = np.vstack([self.x,generator_data])    
        #print(self.y[0].size)
        #self.to_tensor()
    def to_tensor(self):
        self.x = torch.LongTensor(self.x).cuda()
        self.y = torch.LongTensor(self.y).cuda()
        #self.x = torch.LongTensor(self.x).cuda().to(device)
        #self.y = torch.LongTensor(self.y).cuda().to(device)
        #self.x = torch.from_numpy(self.x).float()
        #self.y = torch.from_numpy(self.y).float()
        #self.area = torch.from_numpy(self.area).float()
    def nan_to_zero(self):
        self.x = np.nan_to_num(self.x)
    # Return the target instance (row)
    def __getitem__(self, index_row):
        #self.to_tensor()
        #print(type(self.x))
        return self.x[index_row, :], self.y[index_row]
    # Return the number of instances (the number of rows)
    def __len__(self, dim = 0):
        return len(self.y)

def euclidean_distance(latitude_1, longitude_1, latitude_2, longitude_2):
    return np.sqrt((latitude_1 - latitude_2)**2 + (longitude_1 - longitude_2)**2)
    
"""a = torch.zeros([2,2])
print(a)
b = torch.ones([2,2])
c = torch.cat((a, b), dim = 1)
print(c)"""

#dataset_train = UJIDataset('/home/ubuntu/DLlab/final/UJIndoorLoc/', train = True)
#dataset_test = UJIDataset('/home/ubuntu/final_proj/UJIndoorLoc/', train = False)


#dataset_train[0]


#print(type(dataset_train.x))
#print(len(dataset_train.x))
#print(dataset_train.x)
