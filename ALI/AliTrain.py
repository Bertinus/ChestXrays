from model import *
from AliLoader import *
from ALI_Out import *

import sys
import pickle
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
import time


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
parser.add_argument('--make-check',type = int, default =10000,help="When to make checkpoint")
parser.add_argument('--wrkdir',type = str, default = "NA",help="Output directory of the experiment")
parser.add_argument('--eval', help="No Training",action='store_true',default = False)
parser.add_argument('--inputsize',help="Size of image",default = 32,type=int)
parser.add_argument('--xraydir',help="Directory Chest X-Ray images",default = "./ChestXray-NIHCC-2/",type=str)
parser.add_argument('--seed',help="Random Seed",default = 13,type=int)
parser.add_argument('--verbose',help="Verbose",default = False,action='store_true')
parser.add_argument('--testing',help="Calculate AUCs on other Dataset",default = False,action='store_true')
parser.add_argument('--ToPrint', type=int, default=2500, help='When to print generated sample')
parser.add_argument('--RandomLabel', help="Predicted Label are semi random",default = False,action='store_true')
parser.add_argument('--Restrict',help="Restrict training on this label",default="NA")

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


datadir = opt.xraydir
#ChestXray Image Dir
if os.path.exists("/data/lisa/data/ChestXray-NIHCC-2/"):
    datadir = "/data/lisa/data/ChestXray-NIHCC-2/"

#Print Argument
Params = vars(opt)
Params["ExpDir"] = ExpDir
Params["xraydir"] = datadir
if os.path.exists(ExpDir+"/params.pk"):
    OldParams = pickle.load(open(ExpDir+"/params.pk","rb"))
    for p in OldParams.keys():
        if OldParams[p] != Params[p]:
            print('Warning {0} values are different {1} {2}'.format(p,OldParams[p], Params[p]))
    Req = ["LS","inputsize"]
    for p in Req:
        if OldParams[p] != Params[p]:
            print('Error {0} values are different {1} {2}'.format(p,OldParams[p], Params[p]))
            sys.exit()
    
pickle.dump(Params,open(ExpDir+"/params.pk","wb"))

if opt.verbose:
    print("Loading dataset....")
#Create all the dataset (training and the testing)
dataloader,train_size,test_size,OtherSet,OtherName = CreateDataset(datadir,ExpDir,opt.inputsize,opt.N,batch_size,ModelDir,TestRatio=0.2,rseed=opt.seed,Testing=opt.testing,Restrict=opt.Restrict,MaxSize=10000)

#Keep same random seed for image testing
torch.manual_seed(opt.seed)
ConstantZ = torch.randn(25,LS,1,1)
if torch.cuda.is_available():
    ConstantZ = ConstantZ.cuda()
#GenModel
if opt.verbose:
    print("Loading Models....")
CP = opt.checkpoint #Checkpoint to load (-2 for latest one, -1 for last epoch)
DisX,DisZ,DisXZ,GenZ,GenX,CP,DiscriminatorLoss,AllAUCs = GenModel(opt.inputsize,LS,CP,ExpDir,opt.name,ColorsNumber=ColorsNumber)

#Optimiser
optimizerG = optim.Adam([{'params' : GenX.parameters()},
                         {'params' : GenZ.parameters()}], lr=lr, betas=(b1,b2))

optimizerD = optim.Adam([{'params' : DisZ.parameters()},{'params': DisX.parameters()},
                         {'params' : DisXZ.parameters()}], lr=lr, betas=(b1,b2))

#Loss function
criterion = nn.BCELoss()


if opt.verbose:
    print("Starting Training")
#Training loop!
TotIt = 0
if len(DiscriminatorLoss) > 0:
    TotIt = np.max(list(DiscriminatorLoss.keys()))
print(TotIt)
if Epoch < 0:
    Epoch = CP + (Epoch*-1)+1
cck = 0 #Counter for image
csv = 0 #Counter for saving model
for epoch in range(Epoch):
    #Set to train
    GenX.train()
    GenZ.train()
    DisX.train()
    DisZ.train()
    DisXZ.train()
    
    #What epoch to start from
    #if epoch <= CP:
    #    continue
    
    #Counter per Epoch
    cpt = 0
    
    
    #Some timer
    InitLoadTime = time.time()
    InitTime = time.time()
    
    for dataiter,_ in dataloader:
        
        itime = time.time()
        LoadingTime = (itime - InitLoadTime)
    
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
        
        if opt.RandomLabel == True:
            TrueLabel = (torch.randint(low=70, high=110, size=(1,BS))[0] / 100)
            FakeLabel = (torch.randint(low=-10, high=30, size=(1,BS))[0] / 100)
        
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
        if loss_d > opt.MaxLoss:
            print("Gen is tooooo good:%.2f, No BackProp" % (loss_d.cpu().detach().numpy()+0))
        else:
            optimizerG.zero_grad()
            loss_g.backward()
            optimizerG.step()
    
        #StoreInfo .cpu().numpy()
        
        cpt += BS
        TotIt += BS
        cck += BS
        csv += BS
        DiscriminatorLoss[TotIt] = loss_d.cpu().detach().numpy()+0
        InitLoadTime = time.time()
        RunTime = InitLoadTime - itime
        print("Epoch:%3d c:%6d/%6d = %6.2f Loss:%.4f LoadT=%.4f Rest=%.4f" % (epoch,cpt,train_size,cpt/float(train_size)*100,DiscriminatorLoss[TotIt],LoadingTime,RunTime))
        
        tosave = "%010d" % (TotIt)
        
        if csv > opt.make_check:
            csv = 0
            print("Saving Model")
            # do checkpointing
            torch.save(GenX.state_dict(),
                       '{0}/models/{1}_GenX_It_{2}.pth'.format(ExpDir,opt.name, TotIt))
            torch.save(GenZ.state_dict(),
                       '{0}/models/{1}_GenZ_It_{2}.pth'.format(ExpDir,opt.name, TotIt))
            torch.save(DisX.state_dict(),
                       '{0}/models/{1}_DisX_It_{2}.pth'.format(ExpDir,opt.name, TotIt))
            torch.save(DisZ.state_dict(),
                       '{0}/models/{1}_DisZ_It_{2}.pth'.format(ExpDir,opt.name, TotIt))
            torch.save(DisXZ.state_dict(),
                       '{0}/models/{1}_DisXZ_It_{2}.pth'.format(ExpDir,opt.name, TotIt))
            pickle.dump( DiscriminatorLoss, open( '{0}/models/{1}_Loss_It_{2}.pth'.format(ExpDir,opt.name, TotIt), "wb" ))
            pickle.dump( AllAUCs, open( '{0}/models/{1}_AUCs_It_{2}.pth'.format(ExpDir,opt.name, TotIt), "wb" ))
        if cck > opt.ToPrint:
            cck = 0
            
            #Print some image
        
            #Print generated
            
            #Set to eval
            GenX.eval()
            GenZ.eval()
            DisX.eval()
            DisZ.eval()
            DisXZ.eval()
            
            
            print("Generating Fake")
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
            for i in range(20):
                c +=1
                #print(fd.shape)
                plt.subplot(5,5,c)
                plt.imshow(FakeData[i][0],cmap="gray",vmin=-1,vmax=1)
                plt.title("Fake Disc=%.2f" % (PredFalse[i]))
                plt.axis("off")
            for i in range(5):
                c +=1
                xi = Xnorm[i]
                pi = PredReal[i]
                if torch.cuda.is_available():
                    xi = xi.cpu()
                    pi = pi.cpu()
                xi = xi.detach().numpy()
                pi = pi.detach().numpy()
                plt.subplot(5,5,c)
                plt.imshow(xi[0],cmap="gray",vmin=-1,vmax=1)
                plt.title("Real Disc=%.2f" % (pi))
                plt.axis("off")
            fig.savefig("%s/images/GenImg/GenImg_%s_Gen_epoch_%s.png" % (ExpDir,opt.name,tosave))
            plt.close() 
            
            
            #Calculate AUCs
            if opt.testing == True:
                print("Scoring other DSet")
                AllEvalData = dict()
                for (dl,n) in zip(OtherSet,OtherName):
                    AllEvalData[n] = dict()
                    #Store some value
                    TDiscSc = []
                    TRecErr = []
                    TZ = []
                    TX = []
                    Tlab = []
                    for dataiter,lab in dl:
                        ConstantX = dataiter*2.0-1.0
                        if torch.cuda.is_available():
                            ConstantX = ConstantX.cuda()
                        DiscSc,RecErr,Z = Reconstruct(GenZ,GenX,DisX,DisZ,DisXZ,ConstantX,ExpDir,opt.name,tosave,ImageType = n,Sample = 3,SaveFile=False)
                        TDiscSc += DiscSc
                        TRecErr += RecErr
                        TZ += Z
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
                            
                print("Getting AUCs")
                AllAUCs[TotIt] = dict()
                for n in OtherName:
                    if n == "XRayT":
                        continue
                    tn = "RecLoss"
                    d = AllEvalData[n]["RecLoss"]
                    RealDiscSc = AllEvalData["XRayT"]["RecLoss"]
                    yd = RealDiscSc + d
                    pred = [1]*len(RealDiscSc)+[0]*len(d)
                    fpr, tpr, thresholds = metrics.roc_curve(pred,-np.array(yd), pos_label=1)
                    auc = metrics.auc(fpr, tpr)
                    AllAUCs[TotIt][n] = auc
                    #print(n,auc)
                subdf = pd.DataFrame(AllAUCs).transpose()
                fig = plt.figure(figsize=(8,8))
                for n in list(subdf.columns):
                    plt.plot(subdf.index,subdf[n],label=n,marker="o")
                plt.legend()
                fig.savefig("%s/images/AUCs_LosRec.png" % (ExpDir))
                plt.close() 
            
            #Set to train
            GenX.train()
            GenZ.train()
            DisX.train()
            DisZ.train()
            DisXZ.train()
    
    
print("Saving Model")
# do checkpointing
torch.save(GenX.state_dict(),
           '{0}/models/{1}_GenX_It_{2}.pth'.format(ExpDir,opt.name, TotIt))
torch.save(GenZ.state_dict(),
           '{0}/models/{1}_GenZ_It_{2}.pth'.format(ExpDir,opt.name, TotIt))
torch.save(DisX.state_dict(),
           '{0}/models/{1}_DisX_It_{2}.pth'.format(ExpDir,opt.name, TotIt))
torch.save(DisZ.state_dict(),
           '{0}/models/{1}_DisZ_It_{2}.pth'.format(ExpDir,opt.name, TotIt))
torch.save(DisXZ.state_dict(),
           '{0}/models/{1}_DisXZ_It_{2}.pth'.format(ExpDir,opt.name, TotIt))
pickle.dump( DiscriminatorLoss, open( '{0}/models/{1}_Loss_It_{2}.pth'.format(ExpDir,opt.name, TotIt), "wb" ))
pickle.dump( AllAUCs, open( '{0}/models/{1}_AUCs_It_{2}.pth'.format(ExpDir,opt.name, TotIt), "wb" ))    
            
               
                  
GenX.eval()
GenZ.eval()
DisX.eval()
DisZ.eval()
DisXZ.eval()
print("Generating Fake")
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
for i in range(20):
    c +=1
    #print(fd.shape)
    plt.subplot(5,5,c)
    plt.imshow(FakeData[i][0],cmap="gray",vmin=-1,vmax=1)
    plt.title("Fake Disc=%.2f" % (PredFalse[i]))
    plt.axis("off")
for i in range(5):
    c +=1
    xi = Xnorm[i]
    pi = PredReal[i]
    if torch.cuda.is_available():
        xi = xi.cpu()
        pi = pi.cpu()
    xi = xi.detach().numpy()
    pi = pi.detach().numpy()
    plt.subplot(5,5,c)
    plt.imshow(xi[0],cmap="gray",vmin=-1,vmax=1)
    plt.title("Real Disc=%.2f" % (pi))
    plt.axis("off")
fig.savefig("%s/images/GenImg/GenImg_%s_Gen_epoch_%s.png" % (ExpDir,opt.name,tosave))
plt.close() 
                   
                   
        
       
