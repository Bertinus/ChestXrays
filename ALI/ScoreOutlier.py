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
import time
import argparse



parser = argparse.ArgumentParser()
#Parse command line
parser.add_argument('--LS', type=int, default=128, help='Latent Size')
parser.add_argument('--batch-size', type=int, default=100, help='Batch Size')
parser.add_argument('--name', type=str, default="default", help='Experiment name')
parser.add_argument('--checkpoint', type=int, default=-2, help='Checkpoint epoch to load')
parser.add_argument('--wrkdir',type = str, default = "NA",help="Output directory of the experiment")
parser.add_argument('--inputsize',help="Size of image",default = 32,type=int)
parser.add_argument('--xraydir',help="Directory Chest X-Ray images",default = "./ChestXray-NIHCC-2/",type=str)
parser.add_argument('--verbose',help="Verbose",default = False,action='store_true')

opt = parser.parse_args()

LS = opt.LS #Latent Space Size
batch_size = opt.batch_size #Batch Size
ColorsNumber = 1 #Number of color (always 1 for x-ray)
inputsize = opt.inputsize
datadir = opt.xraydir
isize = opt.inputsize
CP = opt.checkpoint #Checkpoint to load (-2 for latest one, -1 for last epoch)




#ChestXray Image Dir
if os.path.exists("/data/lisa/data/ChestXray-NIHCC-2/"):
    datadir = "/data/lisa/data/ChestXray-NIHCC-2/"
#Create all the folders to save stuff
ExpDir,ModelDir = CreateFolder(opt.wrkdir,opt.name)
#Load data
if not os.path.isfile(ExpDir+"/AllImagesInfo.csv"):
    ImagesInfoDF = ParseXrayCSV(datadir,FileExist=True)
    ImagesInfoDF.to_csv(ExpDir+"/AllImagesInfo.csv")
ImagesInfoDF = pd.read_csv(ExpDir+"/AllImagesInfo.csv")


# Transformations
inputsize = [isize,isize]
data_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(inputsize),
    transforms.ToTensor(),
])

# Initialize dataloader
XrayDataset = XrayDataset(datadir,ImagesInfoDF, transform=data_transforms)
dataloader = DataLoader(XrayDataset, batch_size=batch_size)

#Load model
DisX,DisZ,DisXZ,GenZ,GenX,CP = GenModel(opt.inputsize,LS,CP,ExpDir,opt.name,ColorsNumber=ColorsNumber)

def RecLoss(GenX,GenZ,X):
    GenX.eval()
    GenZ.eval()
    Xr = GenX(GenZ(X))
    DiffX = Xr - X
    if torch.cuda.is_available():
        DiffX = DiffX.cpu()
    DiffX = DiffX.detach().numpy()
    DiffX = np.power(DiffX,2)
    RecLoss = [np.sqrt(np.mean(x)) for x in DiffX]
    return(RecLoss)


AllRecL = []
AllPath = []
GenZ = GenZ.eval()
GenX = GenX.eval()

Count = 0
c = 0
InitLoadTime = time.time()
InitTime = time.time()
for Xi,path in dataloader:
    itime = time.time()
    
    print("Loading Time = %.2f" % ((itime - InitLoadTime) / 60.0))
    
    Xn = Xi*2.0 - 1
    if torch.cuda.is_available():
        Xn = Xn.cuda()
    Rl = RecLoss(GenX,GenZ,Xn)
    AllRecL += Rl
    AllPath += list([p.split("/")[-1] for p in path[0]])
    Count += len(Rl)
    c += len(Rl)
    endtime = time.time()
    TimeLeft = (endtime-InitTime)/float(Count)*float(len(ImagesInfoDF)-Count) / 60.0
    print("%8d/%8d Time=%.4f TimeIm=%.6f MinuteLeft = %.2f" % 
        (Count,len(ImagesInfoDF),(endtime-itime),(endtime-itime) / float(len(Rl)),TimeLeft))
    if c > 1000:
        ErrDF = pd.DataFrame([AllPath,AllRecL]).transpose()
        ErrDF.columns = ["name","RecLoss"]
        ErrDF.to_csv(ExpDir+"/RecLoss.csv")
        c = 0
    InitLoadTime = time.time()
ErrDF = pd.DataFrame([AllPath,AllRecL]).transpose()
ErrDF.columns = ["name","RecLoss"]
ErrDF.to_csv(ExpDir+"/RecLoss.csv")
