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
parser.add_argument('--xraydir',help="Directory Chest X-Ray images",default = "/media/vince/MILA/Chest_data/ChestXray-NIHCC-2/",type=str)
parser.add_argument('--N', type=int, default=-1, help='Number of images to load (-1 for all), default=-1')

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
if opt.N > 0:
    ImagesInfoDF = ImagesInfoDF.head(opt.N)





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
NameDone = []
#Check if already process
if os.path.exists(datadir+"PreProcess/Size"+str(isize)+"/Tensor"+str(isize)+".pt"):
     ImgTensor = torch.load(datadir+"PreProcess/Size"+str(isize)+"/Tensor"+str(isize)+".pt")
     NameDone = list(pd.read_csv(datadir+"PreProcess/Size"+str(isize)+"/AllImagesInfo.csv")["name"])

c  = 0
cpt = 0
InitTime = time.time()

ImagesToDo = ImagesInfoDF[ImagesInfoDF["name"].isin(NameDone) == False]
print("Already Done = %d Left = %d" % (len(NameDone),len(ImagesToDo)))

for fn in list(ImagesToDo["name"].values):
    if fn in NameDone:
        continue
    im = misc.imread(os.path.join(datadir,"images/", fn))
    if len(im.shape) > 2:
        im = im[:, :, 0]
    #Add color chanel
    im = im[:,:,None]
    # Tranform
    im = data_transforms(im)
    im = im.reshape(1,1,isize,isize)
    ImgTensor = torch.cat((ImgTensor, im), 0)
    NameDone.append(fn)
    
    
    #Print some progess
    c += 1
    cpt += 1
    if cpt > len(ImagesInfoDF)/1000.0:
        subdf = ImagesInfoDF[ImagesInfoDF["name"].isin(NameDone)]
    
        torch.save(ImgTensor, datadir+"PreProcess/Size"+str(isize)+"/Tensor"+str(isize)+".pt")
        subdf.to_csv(datadir+"PreProcess/Size"+str(isize)+"/AllImagesInfo.csv")

        cpt = 0
        Fract = c/float(len(ImagesToDo))
        NowTime = time.time()
        Diff = NowTime - InitTime
        Left = Diff / float(c) * float(len(ImagesToDo)-c)
        SecImg = Diff / float(c)
        print("%6d / %6d = %.2f SecImg = %.4f Elaspe Time=%6.2f (min) TimeLeft = %6.2f (min)" % (c,len(ImagesToDo),Fract*100.0,SecImg,Diff/60.0,Left/60.0))
    #print(ImgTensor.shape,im.shape)
        

torch.save(ImgTensor, datadir+"PreProcess/Size"+str(isize)+"/Tensor"+str(isize)+".pt")
subdf = ImagesInfoDF[ImagesInfoDF["name"].isin(NameDone)]
subdf.to_csv(datadir+"PreProcess/Size"+str(isize)+"/AllImagesInfo.csv")







