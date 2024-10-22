from ALImodel import *
from ALIloader import *
from ALImisc import *

from torch.autograd import Variable
import torch.optim as optim
import numpy as np

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
parser.add_argument('--xraydir',help="Directory Chest X-Ray images",default = "/media/vince/MILA/Chest_data/ChestXray-NIHCC-2/",type=str)
parser.add_argument('--seed',help="Random Seed",default = 13,type=int)
parser.add_argument('--verbose',help="Verbose",default = False,action='store_true')
parser.add_argument('--testing',help="Calculate AUCs on other Dataset",default = False,action='store_true')
parser.add_argument('--ToPrint', type=int, default=2500, help='When to print generated sample')
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
if os.path.exists("/network/data1/ChestXray-NIHCC-2/"):
    datadir = "/network/data1/ChestXray-NIHCC-2/"

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
TrainDataset = LoadTrainTestSet(datadir,inputsize,rseed=13,N=Params["N"])
if opt.verbose:
    print("Dataset Len",len(TrainDataset))
#Keep same random seed for image testing
torch.manual_seed(opt.seed)
ConstantZ = torch.randn(40,LS,1,1)
ConstantX = torch.tensor([])
for i in range(10):
    ConstantX = torch.cat((ConstantX, TrainDataset[i][0]*2.0-1.0), 0)
ConstantX = ConstantX.reshape(10,1,inputsize,inputsize)
if torch.cuda.is_available():
    ConstantZ = ConstantZ.cuda()
    ConstantX = ConstantX.cuda()
#GenModel
if opt.verbose:
    print("Loading Models....")
CP = opt.checkpoint #Checkpoint to load (-2 for latest one, -1 for last epoch)
DisX,DisZ,DisXZ,GenZ,GenX,CP,DiscriminatorLoss,AllAUCs = GenModel(opt.inputsize,LS,CP,ExpDir,opt.name,ColorsNumber=ColorsNumber)

#Optimiser
optimizer = optim.Adam([{'params' : GenX.parameters()},
                         {'params' : GenZ.parameters()}], lr=lr, betas=(b1,b2))



#Loss function
criterion = nn.MSELoss()


if opt.verbose:
    print("Starting Training")
    if torch.cuda.is_available():
        print("GPU avaible!")
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
    
    for dataiter,_ in DataLoader(TrainDataset, shuffle=True, batch_size=batch_size,drop_last=True):
        
        itime = time.time()
        LoadingTime = (itime - InitLoadTime)
    
        
        #Get Data
        Xnorm = dataiter *2.0 - 1.0
        #Get Batch Size
        BS = Xnorm.shape[0]
        #Stuff to cuda if cuda
        if torch.cuda.is_available():
            Xnorm = Xnorm.cuda()
        
        
        #Generate Latent from Real
        RealZ = GenZ(Xnorm)
        
        #Generate Fake data from random Latent
        RebuildX = GenX(RealZ)
        
        
        #Get loss 
        loss = criterion(RebuildX, Xnorm)


        #Optimize Discriminator
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        
        cpt += BS
        TotIt += BS
        cck += BS
        csv += BS
        DiscriminatorLoss[TotIt] = loss.cpu().detach().numpy()+0
        InitLoadTime = time.time()
        RunTime = InitLoadTime - itime
        train_size = len(TrainDataset)
        print("Epoch:%3d c:%6d/%6d = %6.2f Loss:%.4f LoadT=%.4f Rest=%.4f" % (epoch,cpt,train_size,cpt/float(train_size)*100,DiscriminatorLoss[TotIt],LoadingTime,RunTime))
        
        tosave = "%010d" % (TotIt)
        
        if csv >= opt.make_check:
            csv = 0
            print("Saving Model")
            # do checkpointing
            SaveModel(GenX,GenZ,DisX,DisZ,DisXZ,DiscriminatorLoss,AllAUCs,ExpDir,opt.name, TotIt)
        
    
    
print("Saving Model")
# do checkpointing
SaveModel(GenX,GenZ,DisX,DisZ,DisXZ,DiscriminatorLoss,AllAUCs,ExpDir,opt.name, TotIt)
                   
                   
        
       
