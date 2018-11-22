from torch.utils.data import Dataset
import glob
import imageio
import os
from scipy import misc
import pandas as pd
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader
import torchvision.datasets as dset

import torch



def CreateDataset(datadir,ExpDir,isize,N,batch_size,ModelDir,TestRatio=0.2,rseed=13,MaxSize = 1000):
  
  #PreProcess folder
  PreProDir = datadir+"PreProcess/Size"+str(isize)
  ImagesInfoDF = pd.read_csv(PreProDir+"/AllImagesInfo.csv")

  #Shuffle the dataset
  np.random.seed(rseed)
  ImagesInfoDF = ImagesInfoDF.sample(frac=1.0,random_state=rseed)

  #Keep number of example
  if N > 0:
      ImagesInfoDF = ImagesInfoDF.head(N)
      
  #Split train and test
  TestSize = int(len(ImagesInfoDF)*TestRatio+0.5)
  if TestSize > MaxSize:
      TestSize = MaxSize
  
  
  
  TestDF = ImagesInfoDF.tail(TestSize)
  TrainDF = ImagesInfoDF.head(len(ImagesInfoDF)-TestSize)
  if not os.path.isfile(ExpDir+"/TrainImagesInfo.csv"):
      
      TrainDF.to_csv(ExpDir+"/TrainImagesInfo.csv")
      TestDF.to_csv(ExpDir+"/TestImagesInfo.csv")
  TrainDF = pd.read_csv(ExpDir+"/TrainImagesInfo.csv")
  TestDF = pd.read_csv(ExpDir+"/TestImagesInfo.csv")
  
  
  print("Train Size = %d Test Size = %d" % (len(TrainDF),len(TestDF)))
  
  train_dataset = XrayDatasetTensor(PreProDir+"/Tensor"+str(isize)+".pt",PreProDir+"/AllImagesInfo.csv",list(TrainDF["name"]))
  test_dataset = XrayDatasetTensor(PreProDir+"/Tensor"+str(isize)+".pt",PreProDir+"/AllImagesInfo.csv",list(TestDF["name"]))

  dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size,drop_last=True)
  ConstantImg = DataLoader(test_dataset, shuffle=False, batch_size=batch_size)
  testing_bs = 100
  #MNIST
  MNIST_transform = transforms.Compose([transforms.Resize(isize),transforms.ToTensor()])
  MNIST_set = dset.MNIST(root=ModelDir, train=True, transform=MNIST_transform, download=True)
  MNIST_loader = DataLoader(dataset=MNIST_set,batch_size=testing_bs,shuffle=False)
  
  data_transforms = transforms.Compose([
      transforms.ToPILImage(),
      transforms.Resize([isize,isize]),
      transforms.ToTensor(),
  ])
  
  #Other Xray
  OtherXRayDir = "./OtherXray/MURA-v1.1/valid/"
  if os.path.exists("/data/lisa/data/MURA-v1.1/MURA-v1.1/train/"):
      OtherXRayDir = "/data/lisa/data/MURA-v1.1/MURA-v1.1/train/"
  OtherXRay = OtherXrayDataset(OtherXRayDir, isize=isize)
  print(len(OtherXRay))
  otherxray = DataLoader(OtherXRay, shuffle=False, batch_size=testing_bs)
  
  

  
  data_transforms = transforms.Compose([
      transforms.ToPILImage(),
      transforms.Resize(isize),
      transforms.RandomVerticalFlip(p=1.0),
      transforms.ToTensor(),
  ])
  #Add Flip
  hflip =  DataLoader(XrayDataset(datadir,TestDF, transform=data_transforms), shuffle=False, batch_size=testing_bs)
  
  data_transforms = transforms.Compose([
      transforms.ToPILImage(),
      transforms.Resize(isize),
      transforms.RandomHorizontalFlip(p=1.0),
      transforms.ToTensor(),
  ])
  #Add Flip
  vflip =  DataLoader(XrayDataset(datadir,TestDF, transform=data_transforms), shuffle=False, batch_size=testing_bs)
  
  
  
  return(dataloader,len(TrainDF),len(TestDF),[ConstantImg,MNIST_loader,otherxray,hflip,vflip],["XRayT","MNIST","OXray","HFlip","VFlip"])

def ParseXrayCSV(datadir,FileExist=False,N=-1):
    lines = [l.rstrip() for l in open(datadir+"Data_Entry_2017.csv")]
    AllDisease = dict()
    ImagesInfo = dict()
    for l in lines[1:]:
        sp = l.split(",")
        if FileExist == True:
            if not os.path.isfile(datadir+"images/"+sp[0]):
                continue
        Tdict = dict()
        Tdict["name"] = sp[0]
        Tdict["patient"] = sp[0].split("_")[0]
        for d in sp[1].split("|"):
            d = d.lower()
            d = d.replace(" ","_")
            AllDisease[d] = 1
            Tdict[d] = 1
        ImagesInfo[sp[0]] = Tdict
        
        if (len(ImagesInfo) >= N) and (N > 0):
            break
    ImagesInfoDF = pd.DataFrame(ImagesInfo).fillna(0).transpose()
    ImagesInfoDF = ImagesInfoDF[["name","patient"]+sorted(AllDisease.keys())]
    return(ImagesInfoDF)



class OtherXrayDataset(Dataset):

    def __init__(self, datadir, isize=None, nrows=-1):

        self.datadir = datadir
        self.ImgFiles = glob.glob(datadir+"/*/*/*/*.png")
        if nrows > 0:
            self.ImgFiles = self.ImgFiles[:nrows]
        types = []
        ImgTensor = torch.tensor([])
        for PathToFile in self.ImgFiles:
            sp = PathToFile.split("/")
            pos = sp[-2].split("_")[-1][0].upper()
            xType = sp[-4].split("_")[-1]
            types.append("_".join([xType,pos]))
            
            
            im = misc.imread(PathToFile)
            if len(im.shape) > 2:
                im = im[:, :, 0]
            #Add color chanel
            im = im[:,:,None]
            # Tranform
            padding = 0
            if im.shape[0] > im.shape[1]:
                padding = (int((im.shape[0]-im.shape[1])/2),0)
            else:
                padding = (0,int((im.shape[1]-im.shape[0])/2))
                
            data_transforms = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Pad(padding,fill=0),
                transforms.Resize([isize,isize]),
                transforms.ToTensor(),
            ])
                
            im = data_transforms(im)
            im = im.reshape(1,1,im.shape[2],im.shape[2])
            ImgTensor = torch.cat((ImgTensor, im), 0)    
            
        self.ImgTensor = ImgTensor    
        self.types = types
        
        #Apply transformation now
        

    def __len__(self):
        return len(self.ImgFiles)

    def __getitem__(self, idx):
        PathToFile = self.ImgFiles[idx]
        types = self.types[idx]
        im = self.ImgTensor[idx]
        
        
        return im,[PathToFile,types]

class XrayDataset(Dataset):

    def __init__(self, datadir,DF, transform=None, nrows=-1):

        self.datadir = datadir
        self.transform = transform
        self.ImgFiles = list(DF["name"].values)
        #print(nrows)
        if nrows > 0:
            self.ImgFiles = self.ImgFiles[:nrows]
            
        ImgTensor = torch.tensor([])
        #Apply transformation now
        for imn in self.ImgFiles:
            PathToFile = os.path.join(self.datadir,"images/", imn)
            im = misc.imread(PathToFile)
            
            if len(im.shape) > 2:
                im = im[:, :, 0]
            #Add color chanel
            im = im[:,:,None]
            # Tranform
            if self.transform:
                im = self.transform(im)
            im = im.reshape(1,1,im.shape[2],im.shape[2])
            ImgTensor = torch.cat((ImgTensor, im), 0)
        self.ImgTensor = ImgTensor

         
    def __len__(self):
        return len(self.ImgFiles)

    def __getitem__(self, idx):
        PathToFile = os.path.join(self.datadir,"images/", self.ImgFiles[idx])
        im = self.ImgTensor[idx]
        return im,[PathToFile,PathToFile]



class XrayDatasetTensor(Dataset):

    def __init__(self, TensorName,FullDF,Names):

        self.ImgTensor = torch.load(TensorName)
        
        df = pd.read_csv(FullDF)
        df.index = list(df["name"])
        tdict =  df[df.columns[4:]].transpose().to_dict()
        self.label = dict()
        for n in tdict.keys():
            tstr = []
            for r in sorted(tdict[n].keys()):
                if tdict[n][r] == 1:
                    tstr.append(r)
            self.label[n] = "_".join(tstr)
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
        
        
        return im,[PathToFile,self.label[self.Names[idx]]]







    
    


