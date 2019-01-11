import argparse
import time
import pickle
from AliMisc import *
from model import *
from AliLoader import *
from sklearn import metrics
from sklearn import manifold

parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str, default="default", help='Experiment name')
parser.add_argument('--wrkdir',type = str, default = "NA",help="Output directory of the experiment")
parser.add_argument('--xraydir',help="Directory Chest X-Ray images",default = "/media/vince/MILA/Chest_data/ChestXray-NIHCC-2/",type=str)
parser.add_argument('--epoch',type=int,help="Epoch to run (-2 run last,-1 run all)",default = -2)
parser.add_argument('--LS', type=int, default=128, help='Latent Size')
parser.add_argument('--inputsize',help="Size of image",default = 32,type=int)

opt = parser.parse_args()

ColorsNumber = 1
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
    Req = ["LS","inputsize"]
    for r in Req:
        Params[r] = OldParams[r]
        
LS = Params["LS"] #Latent Space Size
inputsize = Params["inputsize"]

#Load train and test

TestDataset = LoadTrainTestSet(datadir+"ChestXray-NIHCC-2/",inputsize,rseed=13,subset="Test")
TrainDataset = LoadTrainTestSet(datadir+"ChestXray-NIHCC-2/",inputsize,rseed=13,N=len(TestDataset))
#Load MNIST
MNIST = LoadMNIST(datadir+"MNIST/",inputsize)

#Load MURA
MURA = LoadMURA(datadir+"MURA-v1.1/*/",inputsize,N=len(TestDataset),rseed=13)

#Load Pneunomia
Pneuno = LoadPneunomia(datadir+"/chest_xray/*/*",inputsize,N=len(TestDataset),rseed=13)

#Modified Chest X-ray 
Hflip,Vflip,Shuffle,Random = LoadModChest(datadir+"ChestXray-NIHCC-2/",64,rseed=13)

DsetName = ["ChestXray","tChestXray","MNIST","MURA","Pneuno","hFlip","vFlip","Shuffle","Random"]
Dset = [TestDataset,TrainDataset,MNIST,MURA,Pneuno,Hflip,Vflip,Shuffle,Random]

#Get all Modeled saved
SavedModelsIT = []
for SavedFiles in glob.glob('{0}/models/*_DisXZ_It_*.pth'.format(ExpDir)):
    #print(fck)
    nck = SavedFiles.split("_")[-1].split(".")[0]
    SavedModelsIT.append(int(nck))
SavedModelsIT = sorted(SavedModelsIT)
print("Saved Model",SavedModelsIT)
#Load all data



        
        
