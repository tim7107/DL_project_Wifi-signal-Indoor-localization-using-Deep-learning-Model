import dataloader
import json
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils
import torchvision.transforms as transforms
from torch.autograd import Variable
import argparse
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import random
import numpy as np



import dataloader
import gan
import final
import torch.utils.data as Data
import numpy as np
#----------------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch_size', type=int, default=8, metavar='N',
                    help='input batch size for training (default: 8)')
parser.add_argument('--epochs', type=int, default=50, metavar='N',
                    help='number of epochs to train (default: 164)')
parser.add_argument('--lr', type=float, default=0.002, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--ni', type=int, default=520) # number of input dimension
parser.add_argument('--no', type=int, default=64) # number of output dimension
parser.add_argument('--nc', type=int, default=3) # number of result channel 
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
#-----------------------------------------------------------------------------
encoder = final.Encoder()
decoder = final.Decoder()
classify = final.classify()
if args.cuda:
    device = torch.device('cuda')
    encoder.to(device)
    decoder.to(device)
    classify.to(device)
lr      = args.lr
ratio   = 1.0
encoder_optimizer = optim.Adam(encoder.parameters(), lr=args.lr)
decoder_optimizer = optim.Adam(decoder.parameters(), lr=args.lr)
Batch_size = args.batch_size
Loss = nn.MSELoss()
Loss.requres_grad = True
train_data = dataloader.UJIDataset('/home/ubuntu/DLlab/final/UJIndoorLoc/', train = True,if_gan=1)


#print(len(train_data))
#print((train_data[0][1]))
"""
number_generator = 100
generator_data ,fixed_label= gan.generate(number_generator)
generator_data = generator_data.cpu()
fixed_label    = fixed_label.cpu()
generator_data = generator_data.detach().numpy()
fixed_label    = fixed_label.detach().numpy()

print(generator_data.shape)
print(fixed_label.shape)
temp_1 = tuple(generator_data)
temp_2 = tuple(fixed_label)
temp_3 = tuple(zip(temp_1, temp_2))
#print(temp_3[0][1])



#torch.cat([train_data, generator_data], 1)
"""

train_loader = Data.DataLoader(  
    dataset = train_data,
    batch_size = Batch_size,
    shuffle    = True,
    num_workers = 6
            )    


#------------------------------------------------------------------------------------------------
"""
save_encoder = 1
final.execute(encoder,decoder,save_encoder,train_loader,encoder_optimizer,decoder_optimizer,Loss)
"""
#-----------------------------------------------------------------------------------------------

"""
test_data = dataloader.UJIDataset('/home/ubuntu/DLlab/final/UJIndoorLoc/', train = False , if_gan = False)
test_loader = Data.DataLoader(  
    dataset = test_data,
    batch_size = Batch_size,
    shuffle    = True,
    num_workers = 6
            )




classify_optimizer = optim.Adam(classify.parameters(), lr=args.lr)
path_encoder = '/home/ubuntu/DLlab/final/UJIndoorLoc/weight/encoder.pth'
encoder.load_state_dict(torch.load(path_encoder))
Loss= nn.CrossEntropyLoss()
save_traindata=1
final.traindata(args.epochs,classify,encoder,Loss,save_traindata,train_loader,classify_optimizer,test_loader)
"""


#----------------------------------------------------------------------------------------------



test_data = dataloader.UJIDataset('/home/ubuntu/DLlab/final/UJIndoorLoc/', train = False , if_gan = False)
test_loader = Data.DataLoader(  
    dataset = test_data,
    batch_size = Batch_size,
    shuffle    = True,
    num_workers = 6
            )
Loss= nn.CrossEntropyLoss()
path = '/home/ubuntu/DLlab/final/UJIndoorLoc/weight/encoder.pth'
encoder.load_state_dict(torch.load(path))
classify_path = '/home/ubuntu/DLlab/final/UJIndoorLoc/weight/classify.pth'


classify.load_state_dict(torch.load(classify_path))
final.test(args.epochs,encoder,classify,test_loader,Loss)
