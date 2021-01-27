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
#--------------------------------------settings--------------------------------------------

parser = argparse.ArgumentParser("cDCGAN")
parser.add_argument('--result_dir', type=str, default='result/image')
parser.add_argument('--result_generator', type=str, default='/home/ubuntu/DLlab/final/UJIndoorLoc/weight/')
parser.add_argument('--result_discriminator', type=str, default='result/discriminator')
parser.add_argument('--result_module', type=str, default='result/module')
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--nepoch', type=int, default=50)
parser.add_argument('--nz', type=int, default=100) # number of noise dimension
parser.add_argument('--nc', type=int, default=520) # number of result channel
parser.add_argument('--nfeature', type=int, default=13)
parser.add_argument('--lr', type=float, default=0.002)
#betas = (0.5, 0.99) # adam optimizer beta1, beta2
config, _ = parser.parse_known_args()
#print(config)

#----------------------------------G and D--------------------------------------------------

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(config.nz + 1,256),
            nn.Tanh(),
            nn.Linear(256 , config.nc),
            nn.Tanh(),
        )    
    def forward(self, x, attr):
        #print("x    is " ,x.shape)
        #print("attr is " ,attr.shape)    #attr is  torch.Size([8])
        length = attr.size(0)
        attr = attr.resize(length ,1)
        
        #print("noise shape " , x.shape)               #noise shape  torch.Size([8, 100)
        
        x = torch.cat([x, attr], 1)                    #final shape  torch.Size([8, 101, 1, 1])     
        x = self.main(x)                               #torch.Size([8,520])                     
        return x 
        

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(config.nc + 1,256),
            nn.Tanh(),
            nn.Linear(256 , config.nfeature),
            nn.Tanh(),
            nn.Linear(config.nfeature , 1)
        )
    
    def forward(self, x, attr):
        #print("attr is " ,attr.shape)    #attr is  torch.Size([8])
        #print("x is " ,x.shape)          #x is  torch.Size([8,520])
        length = attr.size(0)
        attr = attr.resize(length ,1)
        x = torch.cat([x, attr], 1)
        #print(x.shape)                  #torch.Size([8, 520+1])
        x = self.main(x)
        return x.view(-1, 1)
        

#-----------------------------------------train---------------------------------------------



class Trainer:
    def __init__(self):
        self.generator = Generator()
        self.discriminator = Discriminator()
        self.loss = nn.MSELoss()
        self.optimizer_g = optim.Adam(self.generator.parameters(), lr=config.lr)
        self.optimizer_d = optim.Adam(self.discriminator.parameters(), lr=config.lr)
        self.generator.cuda()
        self.discriminator.cuda()
        self.loss.cuda()        
        
    def train(self, dataloader):
        noise = Variable(torch.FloatTensor(config.batch_size, config.nz).cuda())    #torch.Size([8, 100, 1, 1])
        label_real = Variable(torch.FloatTensor(config.batch_size, 1).fill_(1).cuda())    #([8,1]) all 1
        label_fake = Variable(torch.FloatTensor(config.batch_size, 1).fill_(0).cuda())    #([8,1]) all 0
        one_hot_labels = torch.FloatTensor(config.batch_size,config.nfeature)             #([8,13]) all 0
        one_hot_labels = one_hot_labels.cuda()
        best_acc=0
        ############## prepare test data######################
        """
        with open('objects.json', 'r') as f:
            data = json.load(f)            
        with open('test.json', 'r') as f:
            test_data = json.load(f)
        
        fixed_noise =  Variable(torch.FloatTensor(32, 100, 1, 1).cuda()) # test noise
        fixed_noise.data.resize_(32, 100, 1, 1).normal_(0, 1)            # normal distribution
        fixed_labels = torch.zeros(32,24)                                # test label

        for i in range(32):
            for j in range(len(test_data[i])):
                #print(test_data[i][j]    #blue cube
                fixed_labels[i,data[test_data[i][j]]]=1
        fixed_labels = Variable(fixed_labels.cuda())
        """
        ############## test data######################
        
        for epoch in range(config.nepoch):
            
            for i, (data, attr) in enumerate(dataloader, 0):
                # train discriminator
                self.discriminator.zero_grad()

                batch_size = data.size(0)
                #print(batch_size)
                label_real = Variable(torch.FloatTensor(batch_size, 1).fill_(1).cuda())
                label_fake = Variable(torch.FloatTensor(batch_size, 1).fill_(0).cuda())
                noise = Variable(torch.FloatTensor(batch_size, config.nz).cuda())
                label_real.data.resize(batch_size, 1).fill_(1)
                label_fake.data.resize(batch_size, 1).fill_(0)
                noise.data.resize_(batch_size, config.nz).normal_(0, 1)
                one_hot_labels.resize_(batch_size, config.nfeature).zero_()
                
                one_hot_labels = attr 
                
            
                real = Variable(data.cuda())   
                one_hot_labels = Variable(one_hot_labels.cuda())
                #print(real.dtype)
                #print(one_hot_labels.dtype)
                one_hot_labels = one_hot_labels.float()
                real =real.float()
                
                for i in range(5):
                    self.generator.zero_grad()
                    fake =   self.generator(noise, one_hot_labels )
                    d_fake = self.discriminator(fake, one_hot_labels )
                    g_loss = self.loss(d_fake, label_real) # trick the fake into being real
                    g_loss.backward(retain_graph=True)
                    self.optimizer_g.step()                
                
                
                d_real = self.discriminator(real, one_hot_labels)

                fake = self.generator(noise, one_hot_labels )
                d_fake = self.discriminator(fake.detach(), one_hot_labels ) # not update generator
                
                #label_real = label_real.long()
                #label_fake = label_fake.long()
                
                d_loss = self.loss(d_real, label_real) + self.loss(d_fake, label_fake) # real label
                d_loss.backward()
                self.optimizer_d.step()

                # train generator

            print("epoch{:03d} d_real: {}, d_fake: {}".format(epoch, d_real.mean(), d_fake.mean()))
            print("epoch{:03d} d_loss: {}, g_loss: {}".format(epoch,d_loss,g_loss))
            #test_fake = self.generator(fixed_noise, fixed_labels)
            #acc=ev.eval(test_fake.data,fixed_labels)
            print("The epoch is {}  ".format(epoch))
            print(" ")
        torch.save(self.generator.state_dict(),'/home/ubuntu/DLlab/final/UJIndoorLoc/weight/generator.pth')
        torch.save(self.discriminator.state_dict(),'/home/ubuntu/DLlab/final/UJIndoorLoc/weight/discriminator.pth')
"""
            if(acc>best_acc):
                torch.save({
                    'epoch': epoch,
                    'generator': self.generator.state_dict(),
                    'discriminator': self.discriminator.state_dict(),
                    'acc': acc,
                    'fixed_noise':fixed_noise,
                    'fixed_labels':fixed_labels,
                },"{}/{}.tar".format(config.result_module,epoch))
                
                torch.save({'state_dict': self.discriminator.state_dict()},
                        '{}/model_d_epoch_{}.pkl'.format(
                            config.result_discriminator,epoch))
                torch.save({'state_dict': self.generator.state_dict()},
                        '{}/model_g_epoch_{}.pkl'.format(
                            config.result_generator, epoch))
                
                best_acc=acc
                
            vutils.save_image(test_fake.data, '{}/result_epoch_{:03d}.png'.format(config.result_dir, epoch), normalize=True)
"""


#------------------------------------------load data and package into dataset---------------------------------------------------------------
"""
Batch_size = config.batch_size
train_data = dataloader.UJIDataset(root = "/home/ubuntu/DLlab/final/UJIndoorLoc/", train = True , if_gan = False)
train_loader     = DataLoader(dataset = train_data, batch_size = Batch_size, num_workers = 4)
"""


"""--------------------------------------------------check everything ----------------------------------------
test_g =  Generator()
test_d =  Discriminator()
test_g.cuda()
test_d.cuda()
noise  =  Variable(torch.FloatTensor(32, config.nz, 1, 1).cuda())
noise.data.resize_(32, config.nz, 1, 1).normal_(0, 1)
for batch_idx, (data, target) in enumerate(train_loader):
        #print(data)
        #print(target.size())      #torch.Size([32, 24])
        #print(data.size())        #torch.Size([32, 3, 64, 64])
        one_hot_labels = torch.FloatTensor(config.batch_size,config.nfeature)
        one_hot_labels = one_hot_labels.cuda()
        one_hot_labels.resize_(32, config.nfeature).zero_()
        one_hot_labels = target  
        data = Variable(data.cuda())  
        target = Variable(target.cuda())
        #test_g(noise , target)
        test_d(data,target)
        if batch_idx > -1:
            break;
"""

#---------------------------------------------------main train-----------------------------------------
if __name__ == '__main__':
    Batch_size = config.batch_size
    train_data = dataloader.UJIDataset(root = "/home/ubuntu/DLlab/final/UJIndoorLoc/", train = True , if_gan = False)
    train_loader     = DataLoader(dataset = train_data, batch_size = Batch_size, num_workers = 4)
    trainer = Trainer()
    trainer.train(train_loader)




#--------------------------------------------------generate module-----------------------------------------


def generate(number_generator):


    fixed_noise =  Variable(torch.FloatTensor(number_generator, 100).cuda()) # test noise
    fixed_noise.data.resize_(number_generator, 100).normal_(0, 1)        # normal distribution

    fixed_label = []
    for i in range(number_generator):
        temp = random.randint(0,12)
        fixed_label.append(temp)
    fixed_label = torch.FloatTensor(fixed_label)
    fixed_label = Variable(fixed_label.cuda())
    fixed_label = fixed_label.resize(number_generator ,1)



    generator = Generator()
    generator.cuda()
    path = "/home/ubuntu/DLlab/final/UJIndoorLoc/weight/generator.pth"

    generator.load_state_dict(torch.load(path))
    generator_data = generator (fixed_noise , fixed_label)
    return generator_data , fixed_label
    
    

#number_generator = 100
#generator_data = generate(number_generator)
#print(generator_data.shape)
#-----------------------------------------------------------------------------------------------------------