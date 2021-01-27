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
import dataloader
import time
import torch.utils.data as Data
import numpy as np
import gan
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#--------------------------------------settings--------------------------------------------


parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch_size', type=int, default=8, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--epochs', type=int, default=50, metavar='N',
                    help='number of epochs to train (default: 164)')
parser.add_argument('--lr', type=float, default=0.002, metavar='LR',
                    help='learning rate (default: 0.005)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--ni', type=int, default=520) # number of input dimension
parser.add_argument('--no', type=int, default=128) # number of output dimension
parser.add_argument('--nc', type=int, default=3) # number of result channel 
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)


#-------------------------------------------------------------------------------------------
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(args.ni,256),
            nn.Tanh(),
            nn.Linear(256 , 128),
            nn.Tanh(),
            nn.Linear(128,args.no),
            nn.Tanh()
        )
    def forward(self, x):
        x= x.float()
        x = self.main(x)
        return x
#-------------------------------------------------------------------------------------------------        
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(args.no,128),
            nn.Tanh(),
            nn.Linear(128,256),
            nn.Tanh(),
            nn.Linear(256,args.ni),
            nn.Sigmoid()
           )
    def forward(self, x):
        x= x.float()
        x = self.main(x)
        x = x.view(x.size(0), -1)
        return x
#-------------------------------------------------------------------------------------------------
class classify(nn.Module):
    def __init__(self):
        super(classify,self).__init__()
        self.main = nn.Sequential(
            nn.Linear(args.no,128),
            nn.Tanh(),
            #nn.Linear(128,13),
            #nn.Tanh()
            )
        self.fc = nn.Linear(128,13)
        self.softmax =nn.Softmax()
    def forward(self,x):
        x= x.float()
        x =self.main(x)
        x = x.view(x.size(0),-1)
        x = self.fc(x)
        x = self.softmax(x)
        return x
#-------------------------------------------------------------------------------------------------

def train(epoch,encoder,decoder,train_loader,encoder_optimizer,decoder_optimizer,Loss):
    encoder.train()
    decoder.train()
    train_loss=0
    correct=0
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.to(device), target.to(device)
        #print(batch_idx)
        answer = data
        target = target.to('cuda').long()   
        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()       
        encoder_output = encoder(data)
        decoder_output = decoder(encoder_output)
        decoder_output = decoder_output.long()               
        predict = decoder_output.long()        
        predict= predict.float()
        answer= answer.float()
        loss = Loss(predict, answer)
        train_loss += loss.data[0]
        #pred = decoder_output.data.max(1)[1]
        #correct += pred.eq(answer.data).cpu().sum()
        loss = loss.requires_grad_()
        loss.backward()
        encoder_optimizer.step()
        decoder_optimizer.step()
        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            epoch, batch_idx * len(data), len(train_loader.dataset),
            100. * batch_idx / len(train_loader), loss))
    temp =correct
    temp = 100. * correct / len(train_loader.dataset)
    return temp
#-------------------------------------------------------------------------------------------------
def traindata(epoch,classify,encoder,Loss,save,train_loader,classify_optimizer,test_loader):
    test_print_acc=[]
    test_print_loss=[]
    best = 0
    for epoch in range(epoch):
        classify.train()
        train_loss=0
        correct= 0

        for batch_idx, (data, target) in enumerate(train_loader):
            #print(type(data))
            if args.cuda:
                data, target = data.to(device), target.to(device)

            encoder_output = encoder(data)
            classify_optimizer.zero_grad()
            classify_output = classify(encoder_output)
            target = target.long()    
            loss = Loss(classify_output , target)
            train_loss += loss.data[0]
            pred = classify_output.data.max(1)[1]
            pred = pred.float()
            target = target.float()
            #print(pred)
            #print(target)
            #print("====================================")
            correct += pred.eq(target.data).cpu().sum()   
            loss = loss.requires_grad_()
            

            
            loss.backward()
            classify_optimizer.step()
        
        test_acc , test_loss = test(epoch,encoder,classify,test_loader,Loss)
        test_print_acc.append(test_acc)
        test_print_loss.append(test_loss)
        
        if(save==1 and test_acc>best):
            torch.save(classify.state_dict(),'./weight/classify.pth')
            best = test_acc
        
        
                            
        print("epoch is ", epoch , "and the coorect is " ,correct.item())
        print("epoch is ", epoch , "and the accuray is " ,correct.item()/len(train_loader.dataset))
        print("==========================")
    if(save==1):
        np.save('UJIindoorLoc_testingdata_loss',test_print_loss)
        np.save('UJIindoorLoc_testingdata_acc',test_print_acc)
#------------------------------------------------------------------------------------------------
def test(epoch,encoder,classify,test_loader,Loss):
    classify.eval()
    encoder.eval()
    test_loss = 0
    correct = 0
    for batch_idx, (data, target) in enumerate(test_loader):
        #data = torch.from_numpy(data)
        if args.cuda:
            data, target = data.to(device), target.to(device)
        target = target.to('cuda').long()
        #with torch.no_grad():
        output = encoder(data)
        output = classify(output)
        target = target.long()              
        test_loss += Loss(output, target).data[0]
        pred = output.data.max(1)[1] # get the index of the max log-probability
        pred = pred.float()
        target = target.float()

        correct += pred.eq(target.data).cpu().sum()

    test_loss = test_loss
    test_loss /= len(test_loader) # loss function already averages over batch size
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.3f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct.item() / len(test_loader.dataset)))
    temp =correct.item()
    temp =100. * correct.item() / len(test_loader.dataset)
    return temp,test_loss     
#-------------------------------------------------------------------------------------------------
def execute(encoder,decoder,save,train_loader,encoder_optimizer,decoder_optimizer,Loss):
    accuracytest = []
    accuracytrain = []
    high=0
    ac=0
    ac_epoch=0
    for epoch in range(1, args.epochs + 1):
        yy=train(epoch,encoder,decoder,train_loader,encoder_optimizer,decoder_optimizer,Loss)
        #xx=test(epoch,net)
        #accuracytest.append(xx)
        #accuracytrain.append(yy)
        #if epoch == 1 :
        #    save(net,string)
        #if xx>high:
        #    high = xx
        #    save(net,string)
        #    print('The high accuracy is',high,'  The epoch is ' ,epoch)
        #    ac = high
        #    ac_epoch=epoch
    if (save==1):
        torch.save(encoder.state_dict(),'./weight/encoder.pth')
    #print('The high accuracy is',ac,'  The epoch is ' ,ac_epoch)
    """         
    plt.title('ResNet18' ,fontsize=14)
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.plot(accuracytest , label= 'ResNet18_'+string+'_' +'testingdata')
    plt.legend(loc='best')
    plt.plot(accuracytrain, label = 'ResGNet18_'+string+'_' +'trainingdata')
    plt.legend(loc='best')
    """
    #np.save('ResNet18_'+string+'_' +'testingdata',accuracytest)
    #np.save('ResNet18_'+string+'_' +'trainingdata',accuracytrain)


#--------------------------------------settings-------------------------------------------
"""
encoder = Encoder()
decoder = Decoder()
classify = classify()
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
train_data = dataloader.UJIDataset('/home/ubuntu/DLlab/final/UJIndoorLoc/', train = True)
train_loader = Data.DataLoader(  
    dataset = train_data,
    batch_size = Batch_size,
    shuffle    = True,
    num_workers = 6
            )
"""
#-------------------------------------------------main----------------------------------------
################          train encoder and decoder               ################
"""
save_encoder = 0
execute(encoder,decoder,save_encoder,train_loader,encoder_optimizer,decoder_optimizer,Loss)
"""


################          train data                              ################
"""
classify_optimizer = optim.Adam(classify.parameters(), lr=args.lr)
path = '/home/ubuntu/DLlab/final/UJIndoorLoc/weight/encoder.pth'
encoder.load_state_dict(torch.load(path))
Loss= nn.CrossEntropyLoss()
save_traindata=0
traindata(args.epochs,classify,encoder,Loss,save_traindata)
"""

################          test data                               ###################
"""
test_data = dataloader.UJIDataset('/home/ubuntu/DLlab/final/UJIndoorLoc/', train = False)
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
test(args.epochs,encoder,classify)
"""