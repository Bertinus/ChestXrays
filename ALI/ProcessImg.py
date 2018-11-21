from AliLoader import *

import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import argparse
import time

print("Lib loaded")
parser = argparse.ArgumentParser()

parser.add_argument('--inputsize',help="Size of image",default = 32,type=int)
parser.add_argument('--xraydir',help="Directory Chest X-Ray images",default = "./ChestXray-NIHCC-2/",type=str)

opt = parser.parse_args()

isize = opt.inputsize
datadir = opt.xraydir

#ChestXray Image Dir
if os.path.exists("/data/lisa/data/ChestXray-NIHCC-2/"):
    datadir = "/data/lisa/data/ChestXray-NIHCC-2/"

#Create Pre-process folder
if not os.path.exists(datadir+"PreProcess"):
    os.makedirs(datadir+"PreProcess")
if not os.path.exists(datadir+"PreProcess/Size"+str(isize)):
    os.makedirs(datadir+"PreProcess/Size"+str(isize))
    
    
#Load data location
if not os.path.isfile(datadir+"/AllImagesInfo.csv"):
    ImagesInfoDF = ParseXrayCSV(datadir,FileExist=True)
    ImagesInfoDF.to_csv(datadir+"/AllImagesInfo.csv")
ImagesInfoDF = pd.read_csv(datadir+"/AllImagesInfo.csv")


#Create Tensor
# Transformations
inputsize = [isize,isize]
data_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(inputsize),
    transforms.ToTensor(),
])

ImgTensor = torch.tensor([])
c  = 0
cpt = 0
InitTime = time.time()
for fn in list(ImagesInfoDF["name"].values):
    im = misc.imread(os.path.join(datadir,"images/", fn))
    if len(im.shape) > 2:
        im = im[:, :, 0]
    #Add color chanel
    im = im[:,:,None]
    # Tranform
    im = data_transforms(im)
    im = im.reshape(1,1,isize,isize)
    ImgTensor = torch.cat((ImgTensor, im), 0)
    

    
    #Print some progess
    c += 1
    cpt += 1
    if cpt > len(ImagesInfoDF)/10.0:
        cpt = 0
        Fract = c/float(len(ImagesInfoDF))
        NowTime = time.time()
        Diff = NowTime - InitTime
        Left = Diff / float(c) * float(len(ImagesInfoDF)-c)
        print("%6d / %6d = %.2f Elaspe Time=%6.2f (min) TimeLeft = %6.2f (min)" % (c,len(ImagesInfoDF),Fract*100.0,Diff/60.0,Left/60.0))
    #print(ImgTensor.shape,im.shape)
        
#torch.save(ImgTensor, datadir+"PreProcess/Size"+str(isize)+"/Tensor"+str(isize)+".pt")
#ImagesInfoDF.to_csv(datadir+"PreProcess/Size"+str(isize)+"/AllImagesInfo.csv")


class XrayDatasetTensor(Dataset):

    def __init__(self, TensorName,FullDF,Names):

        self.ImgTensor = torch.load(TensorName)
        df = pd.read_csv(FullDF)
        self.NameToID = dict()
        for i in range(len(df["name"])):
            self.NameToID[df["name"][i]] = i
        self.Names = Names

    def __len__(self):
        return len(self.Names)

    def __getitem__(self, idx):
        ID = self.NameToID[self.Names[idx]]
        im = self.ImgTensor[ID]
        PathToFile = self.Names[idx]
        print(idx,ID,PathToFile)
        return im,PathToFile
testname = list(ImagesInfoDF.tail()["name"])
print(testname)


dat = XrayDatasetTensor(datadir+"PreProcess/Size"+str(isize)+"/Tensor"+str(isize)+".pt",datadir+"PreProcess/Size"+str(isize)+"/AllImagesInfo.csv",testname)
for i,(img,path) in enumerate(dat):
    print(i,path)




















