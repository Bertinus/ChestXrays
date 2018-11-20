from model import *
from AliLoader import *
from ALI_Out import *

from torch.utils.data import DataLoader
import torchvision.datasets as dset
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import manifold
from sklearn import metrics
from scipy import stats
from AliMisc import *
import argparse



parser = argparse.ArgumentParser()
#Parse command line
parser.add_argument('--epoch', type=int, default=100, help='Epoch')
parser.add_argument('--LS', type=int, default=128, help='Latent Size')
parser.add_argument('--batch-size', type=int, default=100, help='Batch Size')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--beta2', type=float, default=1e-3, help='beta2 for adam. default=0.999')
parser.add_argument('--MaxLoss', type=float, default=999.0, help='MaxLoss Default=999')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate for optimizer, default=0.00005')
parser.add_argument('--N', type=int, default=-1, help='Number of images to load (-1 for all), default=-1')
parser.add_argument('--name', type=str, default="default", help='Experiment name')
parser.add_argument('--checkpoint', type=int, default=-3, help='Checkpoint epoch to load')
parser.add_argument('--make-check',type = int, default =1,help="When to make checkpoint")
parser.add_argument('--wrkdir',type = str, default = "NA",help="Output directory of the experiment")
parser.add_argument('--eval', help="No Training",action='store_true',default = False)
parser.add_argument('--inputsize',help="Size of image",default = 32,type=int)
parser.add_argument('--xraydir',help="Directory Chest X-Ray images",default = "./ChestXray-NIHCC-2/",type=str)
parser.add_argument('--seed',help="Random Seed",default = 13,type=int)
parser.add_argument('--verbose',help="Verbose",default = False,action='store_true')

opt = parser.parse_args()


Epoch = opt.epoch #Number of Epoch
LS = opt.LS #Latent Space Size
batch_size = opt.batch_size #Batch Size
ColorsNumber = 1 #Number of color (always 1 for x-ray)
inputsize = opt.inputsize
lr = opt.lr
b1 = opt.beta1
b2 = opt.beta2

#Create all the folders to save stuff
ExpDir,ModelDir = CreateFolder(opt.wrkdir,opt.name)

#Print Argument
print("lr=%f" % (lr))
print("b1=%f" % (b1))
print("b2=%f" % (b2))
print("batch_size=%d" % (batch_size))
print("inputsize=%d" % (inputsize))
print("seed=%d" % (opt.seed))
print("name=%s" % (name))



datadir = opt.xraydir
#ChestXray Image Dir
if os.path.exists("/data/lisa/data/ChestXray-NIHCC-2/"):
    datadir = "/data/lisa/data/ChestXray-NIHCC-2/"


if opt.verbose:
    print("Loading dataset....")
#Create all the dataset (training and the testing)
dataloader,train_size,test_size,OtherSet,OtherName = CreateDataset(datadir,ExpDir,opt.inputsize,opt.N,batch_size,ModelDir,TestRatio=0.2,rseed=opt.seed)

#Keep same random seed for image testing  
ConstantZ = torch.randn(9,LS,1,1)
if torch.cuda.is_available():
    ConstantZ = ConstantZ.cuda()
#GenModel
if opt.verbose:
    print("Loading Models....")
CP = opt.checkpoint #Checkpoint to load (-2 for latest one, -1 for last epoch)
DisX,DisZ,DisXZ,GenZ,GenX,CP = GenModel(opt.inputsize,LS,CP,ExpDir,opt.name,ColorsNumber=ColorsNumber)





sys.exit()
#Optimiser
optimizerG = optim.Adam([{'params' : GenX.parameters()},
                         {'params' : GenZ.parameters()}], lr=lr, betas=(b1,b2))

optimizerD = optim.Adam([{'params' : DisZ.parameters()},{'params': DisX.parameters()},
                         {'params' : DisXZ.parameters()}], lr=lr, betas=(b1,b2))

#Loss function
criterion = nn.BCELoss()
DiscriminatorLoss = []

if opt.verbose:
    print("Starting Training")
#Training loop!
for epoch in range(Epoch):
    #Set to train
    GenX.train()
    GenZ.train()
    DisX.train()
    DisZ.train()
    DisXZ.train()
    
    #What epoch to start from
    if epoch <= CP:
        continue
    
    #Counter per Epoch
    cpt = 0
    for dataiter in dataloader:
        #If only evaluating don't train!
        if opt.eval == True:
            break
        #Get Data
        Xnorm = dataiter *2.0 - 1.0
        #Get Batch Size
        BS = Xnorm.shape[0]
        #Ignore weirdly small batchsize
        #if BS < batch_size/10.0:
        #    continue
        
        #Generate Random latent variable
        FakeZ = torch.randn(BS,LS,1,1)
        
        #Stuff to cuda if cuda
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
        TrueLabel = Variable(torch.ones(BS)-0.1) #-0.1 is a trick from the internet
        FakeLabel = Variable(torch.zeros(BS))
        if torch.cuda.is_available():
            TrueLabel = TrueLabel.cuda()
            FakeLabel = FakeLabel.cuda()
        #Get loss for discriminator
        loss_d = criterion(PredReal.view(-1), TrueLabel) + criterion(PredFalse.view(-1), FakeLabel)

        #Get loss for generator
        loss_g = criterion(PredFalse.view(-1), TrueLabel) + criterion(PredReal.view(-1), FakeLabel)

        #Optimize Discriminator
        if loss_d < opt.MaxLoss:
            optimizerD.zero_grad()
            loss_d.backward(retain_graph=True)
            optimizerD.step()
        else:
            print("Disc is tooooo good:%.2f" % (loss_d.cpu().detach().numpy()+0))
        #Optimize Generator
        optimizerG.zero_grad()
        loss_g.backward()
        optimizerG.step()
    
        #StoreInfo .cpu().numpy()
        DiscriminatorLoss.append(loss_d.cpu().detach().numpy()+0)
        cpt += BS
        print("Epoch:%3d c:%6d/%6d = %6.2f Loss:%.4f" % (epoch,cpt,train_size,cpt/float(train_size)*100,DiscriminatorLoss[-1]))
    tosave = -1
    tosaveint = -1
    if epoch % opt.make_check == 0:
        tosave = "%06d" % (epoch)
        tosaveint = epoch
    if opt.eval == False:
        # do checkpointing
        torch.save(GenX.state_dict(),
                   '{0}/models/{1}_GenX_epoch_{2}.pth'.format(ExpDir,opt.name, tosaveint))
        torch.save(GenZ.state_dict(),
                   '{0}/models/{1}_GenZ_epoch_{2}.pth'.format(ExpDir,opt.name, tosaveint))
        torch.save(DisX.state_dict(),
                   '{0}/models/{1}_DisX_epoch_{2}.pth'.format(ExpDir,opt.name, tosaveint))
        torch.save(DisZ.state_dict(),
                   '{0}/models/{1}_DisZ_epoch_{2}.pth'.format(ExpDir,opt.name, tosaveint))
        torch.save(DisXZ.state_dict(),
                   '{0}/models/{1}_DisXZ_epoch_{2}.pth'.format(ExpDir,opt.name, tosaveint))
               
               
    #Print some image
    
    #Print generated
    
    #Set to eval
    GenX.eval()
    GenZ.eval()
    DisX.eval()
    DisZ.eval()
    DisXZ.eval()
    
    
    
    #Print Fake
    with torch.no_grad():
        FakeData = GenX(ConstantZ)
        PredFalse = DisXZ(torch.cat((DisZ(ConstantZ), DisX(FakeData)), 1))

        if torch.cuda.is_available():
            FakeData = FakeData.cpu()
            PredFalse = PredFalse.cpu()
        
        FakeData = FakeData.detach().numpy()
        PredFalse= PredFalse.detach().numpy()
    
    fig = plt.figure(figsize=(8,8))
    c = 0
    for i in range(9):
        c +=1
        #print(fd.shape)
        plt.subplot(3,3,c)
        plt.imshow(FakeData[i][0],cmap="gray",vmin=-1,vmax=1)
        plt.title("Disc=%.2f" % (PredFalse[i]))
        plt.axis("off")
    fig.savefig("%s/images/%s_Gen_epoch_%s.png" % (ExpDir,opt.name,tosave))
    
    AllEvalData = dict()
    for (dl,n) in zip(OtherSet,OtherName):
        AllEvalData[n] = dict()
        toprint = True
        
        #Store some value
        TDiscSc = []
        TRecErr = []
        TZ = []
        TX = []
        for dataiter in dl:
            if n == "MNIST":
                dataiter = dataiter[0]
            ConstantX = dataiter*2.0-1.0
            if torch.cuda.is_available():
                ConstantX = ConstantX.cuda()
            DiscSc,RecErr,Z = Reconstruct(GenZ,GenX,DisX,DisZ,DisXZ,ConstantX,ExpDir,opt.name,tosave,ImageType = n,Sample = 3,SaveFile=toprint)
            TDiscSc += DiscSc
            TRecErr += RecErr
            TZ += Z
            
            #Keep image
            if torch.cuda.is_available():
                ConstantX = ConstantX.cpu()
            TX += list(ConstantX.detach().numpy())
            #print(TX)
            toprint = False
            if len(TZ) > test_size:
                TZ = TZ[:test_size]
                TX = TX[:test_size]
                TDiscSc = TDiscSc[:test_size]
                TRecErr = TRecErr[:test_size]
                break
        AllEvalData[n]["Z"] = TZ
        AllEvalData[n]["X"] = TX
        AllEvalData[n]["RecLoss"] = TRecErr
        AllEvalData[n]["Dis"] = TDiscSc
    
    
    
    for n in ["XRayT"]:
        c = 0
        fig = plt.figure(figsize=(8,8))
        sind = np.argsort(AllEvalData[n]["RecLoss"])
        if n == "XRayT":
            sind = sind[::-1]
        
        for ind in sind[0:9]:
            c +=1
            plt.subplot(3,3,c)
            plt.imshow(AllEvalData[n]["X"][ind][0],cmap="gray",vmin=-1,vmax=1)
            
            plt.title("RecLoss=%.2f" % (AllEvalData[n]["RecLoss"][ind]))
            plt.axis("off")
        fig.savefig("%s/images/%s_%s_SortError_epoch_%s.png" % (ExpDir,opt.name,n,tosave))
    
    #Do T-SNE
    AllZ = []
    for n in sorted(AllEvalData.keys()):
        AllZ += AllEvalData[n]["Z"]
    Y = manifold.TSNE(n_components=2).fit_transform(AllZ)
    fig = plt.figure()
    minc = 0
    maxc = 0
    for n in sorted(AllEvalData.keys()):
      maxc += len(AllEvalData[n]["Z"])
      plt.scatter(Y[minc:maxc,0],Y[minc:maxc,1],label=n)
      minc = maxc
    plt.legend()
    fig.savefig("%s/images/%s_TSNE_epoch_%s.png" % (ExpDir,opt.name,tosave))
    
    if opt.eval == True:
        break
        
                   
                   
        
       
