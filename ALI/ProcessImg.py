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
parser.add_argument('--N', type=int, default=-1, help='Number of images to load (-1 for all), default=-1')
parser.add_argument('--seed', type=int, default=13, help='RandomSeed')
parser.add_argument('--MaxTest', type=int, default=999999, help='Maximum Test Set size (default 999999)')
parser.add_argument('--split', type=float, default=0.8, help='Train / Test Split (default=0.8)')

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
    print("Creating Dataframe")
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
print("Converting")
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
    if cpt > len(ImagesInfoDF)/1000.0:
        torch.save(ImgTensor, datadir+"PreProcess/Size"+str(isize)+"/Tensor"+str(isize)+".pt")
        ImagesInfoDF.to_csv(datadir+"PreProcess/Size"+str(isize)+"/AllImagesInfo.csv")


        cpt = 0
        Fract = c/float(len(ImagesInfoDF))
        NowTime = time.time()
        Diff = NowTime - InitTime
        Left = Diff / float(c) * float(len(ImagesInfoDF)-c)
        SecImg = Diff / float(c)
        print("%6d / %6d = %.2f SecImg = %.4f Elaspe Time=%6.2f (min) TimeLeft = %6.2f (min)" % (c,len(ImagesInfoDF),Fract*100.0,SecImg,Diff/60.0,Left/60.0))
    #print(ImgTensor.shape,im.shape)
        

torch.save(ImgTensor, datadir+"PreProcess/Size"+str(isize)+"/Tensor"+str(isize)+".pt")
ImagesInfoDF.to_csv(datadir+"PreProcess/Size"+str(isize)+"/AllImagesInfo.csv")







