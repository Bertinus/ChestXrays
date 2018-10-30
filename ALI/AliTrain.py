from model import *
from AliLoader import *
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.optim as optim
import numpy as np

import argparse

parser = argparse.ArgumentParser()
#Parse command line
parser.add_argument('--epoch', type=int, default=10, help='Epoch')
parser.add_argument('--LS', type=int, default=128, help='Latent Size')
parser.add_argument('--batch-size', type=int, default=100, help='Batch Size')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--beta2', type=float, default=1e-3, help='beta2 for adam. default=0.999')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate for optimizer, default=0.00005')
parser.add_argument('--N', type=int, default=-1, help='Number of images to load (-1 for all), default=-1')
parser.add_argument('--name', type=str, default="default", help='Experiment name')
parser.add_argument('--checkpoint', type=int, default=-2, help='Checkpoint epoch to load')
parser.add_argument('--make-check',type = int, default =5,help="When to make checkpoint")

opt = parser.parse_args()
print(opt)

Epoch = opt.epoch
LS = opt.LS #Latent Space Size
batch_size = opt.batch_size
ColorsNumber = 1

lr = opt.lr
b1 = opt.beta1
b2 = opt.beta2

#Directory with all model
ModelDir = "./model/"
if os.path.exists("/data/milatmp1/frappivi/ALI_model"):
    ModelDir = "/data/milatmp1/frappivi/ALI_model/"
    
ExpDir = ModelDir+opt.name
if not os.path.exists(ExpDir):
    os.makedirs(ExpDir)
    os.makedirs(ExpDir+"/models")
    os.makedirs(ExpDir+"/images")
    
    
    
print("Wrkdir = %s" % (ExpDir))



#Encoder param
EncKernel = [5,4,4,4,4,1,1]
EncStride = [1,2,1,2,1,1,1]
EncDepth = [32,64,128,256,512,512,LS]

#Generator param
GenKernel = [4,4,4,4,5,1,1]
GenStride = [1,2,1,2,1,1,1]
GenDepth = [256,128,64,32,32,32,ColorsNumber]

#Discriminator X param
DxKernel = [5,4,4,4,4]
DxStride = [1,2,1,2,1]
DxDepth = [32,64,128,256,512]

#Discriminator Z param
DzKernel = [1,1]
DzStride = [1,1]
DzDepth = [512,512]

#Concat Discriminator param
DxzKernel = [1,1,1]
DxzStride = [1,1,1]
DxzDepth = [1024,1024,1]




#Image Dir
datadir = "./images/"
if os.path.exists("/data/lisa/data/ChestXray-NIHCC-2/images"):
    datadir = "/data/lisa/data/ChestXray-NIHCC-2/images/"



#Load data
# Transformations
inputsize = [224, 224]
inputsize = [32,32]
data_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(inputsize),
    transforms.ToTensor(),
    #transforms.Lambda(lambda x: x.repeat(3, 1, 1))
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Initialize dataloader
dataset = XrayDataset(datadir, transform=data_transforms,nrows=opt.N)
dataloader = DataLoader(dataset, shuffle=True, batch_size=batch_size)
DataLen = len(dataset)
print("Dataset Len = %d" % (DataLen))

#Create Model

DisX = DiscriminatorX(KS=DxKernel,ST=DxStride,DP=DxDepth)
DisZ = DiscriminatorZ(KS=DzKernel,ST=DzStride,DP=DzDepth,LS=LS)
DisXZ = DiscriminatorXZ(KS=DxzKernel,ST=DxzStride,DP=DxzDepth)

GenZ = Encoder(KS=EncKernel,ST=EncStride,DP=EncDepth,LS=LS)
GenX = Generator(latent_size=LS,KS=GenKernel,ST=GenStride,DP=GenDepth)

#Check if checkpoint exist
if os.path.isfile('{0}/models/{1}_DisXZ_epoch_{2}.pth'.format(ExpDir,opt.name, opt.checkpoint)):
    print("Checkpoint %d exist, will load param and start training from there" % (opt.checkpoint))
    DisX.load_state_dict(torch.load('{0}/models/{1}_DisX_epoch_{2}.pth'.format(ExpDir,opt.name, opt.checkpoint)))
    DisZ.load_state_dict(torch.load('{0}/models/{1}_DisZ_epoch_{2}.pth'.format(ExpDir,opt.name, opt.checkpoint)))
    DisXZ.load_state_dict(torch.load('{0}/models/{1}_DisXZ_epoch_{2}.pth'.format(ExpDir,opt.name, opt.checkpoint)))
    
    GenZ.load_state_dict(torch.load('{0}/models/{1}_GenZ_epoch_{2}.pth'.format(ExpDir,opt.name, opt.checkpoint)))
    GenX.load_state_dict(torch.load('{0}/models/{1}_GenX_epoch_{2}.pth'.format(ExpDir,opt.name, opt.checkpoint)))
    print("Done loading")

if torch.cuda.is_available():
    print("cuda is available")
    DisX = DisX.cuda()
    DisZ = DisZ.cuda()
    DisXZ = DisXZ.cuda()

    GenZ = GenZ.cuda()
    GenX = GenX.cuda()

#Write a bunch of param about training run




optimizerG = optim.Adam([{'params' : GenX.parameters()},
                         {'params' : GenZ.parameters()}], lr=lr, betas=(b1,b2))

optimizerD = optim.Adam([{'params' : DisZ.parameters()},{'params': DisX.parameters()},
                         {'params' : DisXZ.parameters()}], lr=lr, betas=(b1,b2))

DiscriminatorLoss = []

from torch.autograd import Variable


ConstantZ = torch.randn(9,LS,1,1)

cpt = 0

criterion = nn.BCELoss()
for epoch in range(Epoch):
    if epoch <= opt.checkpoint:
        continue
    c = 0
    for dataiter in dataloader:
        
        #Get Data
        Xnorm = dataiter *2.0 - 1.0
        #Get Batch Size
        BS = Xnorm.shape[0]
        #Ignore weirdly small batchsize
        if BS < batch_size/2.0:
            continue
        
        FakeZ = torch.randn(BS,LS,1,1)
        #To cuda
        if torch.cuda.is_available():
            Xnorm = Xnorm.cuda()
            FakeZ = FakeZ.cuda()
        
        
        
        #Generate Fake data from random Latent
        
        FakeX = GenX(FakeZ)
        
        #Generate Latent from Real
        RealZ = GenZ(Xnorm)
        
        #Have discriminator do is thing on real and fake data
        RealCat= torch.cat((DisZ(RealZ), DisX(Xnorm)), 1)
        FakeCat= torch.cat((DisZ(FakeZ), DisX(FakeX)), 1)
        PredReal  = DisXZ(RealCat)
        PredFalse = DisXZ(FakeCat)
        
        
        #Gen fake and true label
        TrueLabel = Variable(torch.ones(BS)-0.1)
        FakeLabel = Variable(torch.zeros(BS))
        if torch.cuda.is_available():
            TrueLabel = TrueLabel.cuda()
            FakeLabel = FakeLabel.cuda()
        #Get loss for discriminator
        loss_d = criterion(PredReal.view(-1), TrueLabel) + criterion(PredFalse.view(-1), FakeLabel)

        #Get loss for generator
        loss_g = criterion(PredFalse.view(-1), TrueLabel) + criterion(PredReal.view(-1), FakeLabel)

        #Optimize Discriminator
        optimizerD.zero_grad()
        loss_d.backward(retain_graph=True)
        optimizerD.step()
    
        #Optimize Generator
        optimizerG.zero_grad()
        loss_g.backward()
        optimizerG.step()
    
        #StoreInfo .cpu().numpy()
        DiscriminatorLoss.append(loss_d.cpu().detach().numpy()+0)
        c += BS
        print("Epoch:%3d c:%6d/%6d = %.2f Loss:%.4f" % (epoch,c,DataLen,c/float(DataLen)*100,DiscriminatorLoss[-1]))
        
    tosave = -1
    if epoch % opt.make_check == 0:
        tosave = epoch    
    # do checkpointing
    torch.save(GenX.state_dict(),
               '{0}/models/{1}_GenX_epoch_{2}.pth'.format(ExpDir,opt.name, tosave))
    torch.save(GenZ.state_dict(),
               '{0}/models/{1}_GenZ_epoch_{2}.pth'.format(ExpDir,opt.name, tosave))
    torch.save(DisX.state_dict(),
               '{0}/models/{1}_DisX_epoch_{2}.pth'.format(ExpDir,opt.name, tosave))
    torch.save(DisZ.state_dict(),
               '{0}/models/{1}_DisZ_epoch_{2}.pth'.format(ExpDir,opt.name, tosave))
    torch.save(DisXZ.state_dict(),
               '{0}/models/{1}_DisXZ_epoch_{2}.pth'.format(ExpDir,opt.name, tosave))
