from model import *
from AliLoader import *
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.optim as optim
import numpy as np

Epoch = 10
LS = 128 #Latent Space Size
batch_size = 24
ColorsNumber = 1

lr = 1e-4
b1 = 0.5
b2 = 1e-3

ModelDir = "./model"

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
dataset = XrayDataset(datadir, transform=data_transforms)
dataloader = DataLoader(dataset, shuffle=True, batch_size=batch_size)


#Create Model

DisX = DiscriminatorX(KS=DxKernel,ST=DxStride,DP=DxDepth)
DisZ = DiscriminatorZ(KS=DzKernel,ST=DzStride,DP=DzDepth,LS=LS)
DisXZ = DiscriminatorXZ(KS=DxzKernel,ST=DxzStride,DP=DxzDepth)

GenZ = Encoder(KS=EncKernel,ST=EncStride,DP=EncDepth,LS=LS)
GenX = Generator(latent_size=LS,KS=GenKernel,ST=GenStride,DP=GenDepth)

if torch.cuda.is_available():
    print("cuda is avaible")
    DisX = DiscriminatorX(KS=DxKernel,ST=DxStride,DP=DxDepth).cuda()
    DisZ = DiscriminatorZ(KS=DzKernel,ST=DzStride,DP=DzDepth,LS=LS).cuda()
    DisXZ = DiscriminatorXZ(KS=DxzKernel,ST=DxzStride,DP=DxzDepth).cuda()

    GenZ = Encoder(KS=EncKernel,ST=EncStride,DP=EncDepth,LS=LS).cuda()
    GenX = Generator(latent_size=LS,KS=GenKernel,ST=GenStride,DP=GenDepth).cuda()




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
    c = 0
    for dataiter in dataloader:
        c += 1
        #Get Data
        Xnorm = dataiter *2.0 - 1.0
        #To cuda
        if torch.cuda.is_available():
            Xnorm = Xnorm.cuda()
        
        #Get Batch Size
        BS = Xnorm.shape[0]
        if BS < batch_size/2.0:
            continue
        
        #Generate Fake data from random Latent
        FakeZ = torch.randn(BS,LS,1,1)
        FakeX = GenX(FakeZ)
        
        #Generate Latent from Real
        RealZ = GenZ(Xnorm)
        
        #Have discriminator do is thing on real and fake data
        RealCat= torch.cat((DisZ(RealZ), DisX(Xnorm)), 1)
        FakeCat= torch.cat((DisZ(FakeZ), DisX(FakeX)), 1)
        PredReal  = DisXZ(RealCat)
        PredFalse = DisXZ(FakeCat)
        
        #Get loss for discriminator
        loss_d = criterion(PredReal.view(-1), Variable(torch.ones(BS)-0.1)) + criterion(PredFalse.view(-1), Variable(torch.zeros(BS)))

        #Get loss for generator
        loss_g = criterion(PredFalse.view(-1), Variable(torch.ones(BS)-0.1)) + criterion(PredReal.view(-1), Variable(torch.zeros(BS)))

        #Optimize Discriminator
        optimizerD.zero_grad()
        loss_d.backward(retain_graph=True)
        optimizerD.step()
    
        #Optimize Generator
        optimizerG.zero_grad()
        loss_g.backward()
        optimizerG.step()
    
        #StoreInfo
        DiscriminatorLoss.append(loss_d.detach().numpy()+0)

        print("Epoch:%d c:%d Loss:%.4f" % (epoch,c,DiscriminatorLoss[-1]))
        
        
        
    # do checkpointing
    torch.save(GenX.state_dict(),
               '{0}/GenX_epoch_{1}.pth'.format(ModelDir, epoch))
    torch.save(GenZ.state_dict(),
               '{0}/GenZ_epoch_{1}.pth'.format(ModelDir, epoch))
    torch.save(DisX.state_dict(),
               '{0}/DisX_epoch_{1}.pth'.format(ModelDir, epoch))
    torch.save(DisZ.state_dict(),
               '{0}/DisZ_epoch_{1}.pth'.format(ModelDir, epoch))
    torch.save(DisXZ.state_dict(),
               '{0}/DisXZ_epoch_{1}.pth'.format(ModelDir, epoch))
