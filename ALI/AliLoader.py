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





def CreateDataset(datadir,ExpDir,isize,N,batch_size,ModelDir,TestRatio=0.2,rseed=13):
  
  
  #Load data
  if not os.path.isfile(ExpDir+"/AllImagesInfo.csv"):
      ImagesInfoDF = ParseXrayCSV(datadir,FileExist=True)
      ImagesInfoDF.to_csv(ExpDir+"/AllImagesInfo.csv")
  ImagesInfoDF = pd.read_csv(ExpDir+"/AllImagesInfo.csv")


  
  UniID = np.unique(ImagesInfoDF["patient"])
  np.random.seed(13)

  np.random.shuffle(UniID)
  test_size = int(len(UniID)*TestRatio+0.5)
  if test_size > 1000:
      test_size = 1000
  train_size = int(len(UniID)-test_size)


  TrainID = UniID[:train_size]
  TestID = UniID[train_size:]
  
  if not os.path.isfile(ExpDir+"/TrainImagesInfo.csv"):
  
      TestDF = ImagesInfoDF[ImagesInfoDF["patient"].isin(TestID)].sample(frac=1.0, random_state=13)
      TrainDF = ImagesInfoDF[ImagesInfoDF["patient"].isin(TrainID)].sample(frac=1.0, random_state=13)
      if N > -1:
          TrainDF = TrainDF.head(N)
      if len(TestDF) > len(TrainDF)/5.0:
          TestDF = TestDF.head(int(len(TrainDF)/5.0))

      if len(TestDF) > 5000:
          TestDF = TestDF.head(5000)
      TrainDF.to_csv(ExpDir+"/TrainImagesInfo.csv")
      TestDF.to_csv(ExpDir+"/TestImagesInfo.csv")
  TrainDF = pd.read_csv(ExpDir+"/TrainImagesInfo.csv")
  TestDF = pd.read_csv(ExpDir+"/TestImagesInfo.csv")

  # Transformations
  inputsize = [isize,isize]
  data_transforms = transforms.Compose([
      transforms.ToPILImage(),
      transforms.Resize(inputsize),
      transforms.ToTensor(),
  ])

  # Initialize dataloader
  train_dataset = XrayDataset(datadir,TrainDF, transform=data_transforms)
  test_dataset =  XrayDataset(datadir,TestDF, transform=data_transforms)
  test_size = len(TestDF)
  train_size = len(TrainDF)    
  testing_bs = 500
  if testing_bs > len(TestDF):
      testing_bs = len(TestDF)
  print("Test Size = %d Train Size = %d" % (len(TestDF),len(TrainDF)))

  dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
  ConstantImg = DataLoader(test_dataset, shuffle=False, batch_size=batch_size)

  #MNIST
  MNIST_transform = transforms.Compose([transforms.Resize(inputsize),transforms.ToTensor()])
  MNIST_set = dset.MNIST(root=ModelDir, train=True, transform=MNIST_transform, download=True)
  MNIST_loader = DataLoader(dataset=MNIST_set,batch_size=testing_bs,shuffle=False)

  #Other Xray
  OtherXRayDir = "/data/lisa/data/MURA-v1.1/MURA-v1.1/train/"
  OtherXRayDir = "./OtherXray/"
  OtherXRay = OtherXrayDataset(OtherXRayDir, transform=data_transforms)
  otherxray = DataLoader(OtherXRay, shuffle=False, batch_size=testing_bs)
  
  

  
  data_transforms = transforms.Compose([
      transforms.ToPILImage(),
      transforms.Resize(inputsize),
      transforms.RandomVerticalFlip(p=1.0),
      transforms.ToTensor(),
  ])
  #Add Flip
  hflip =  DataLoader(XrayDataset(datadir,TestDF, transform=data_transforms), shuffle=False, batch_size=testing_bs)
  
  data_transforms = transforms.Compose([
      transforms.ToPILImage(),
      transforms.Resize(inputsize),
      transforms.RandomHorizontalFlip(p=1.0),
      transforms.ToTensor(),
  ])
  #Add Flip
  vflip =  DataLoader(XrayDataset(datadir,TestDF, transform=data_transforms), shuffle=False, batch_size=testing_bs)
  
  
  
  return(dataloader,train_size,test_size,[ConstantImg,MNIST_loader,otherxray,hflip,vflip],["XRayT","MNIST","OXray","HFlip","VFlip"])

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

    def __init__(self, datadir, transform=None, nrows=-1):

        self.datadir = datadir
        self.transform = transform
        self.ImgFiles = [f.split(datadir)[-1] for f in glob.glob(datadir+"/*/*/*/*.png")]
        if nrows > 0:
            self.ImgFiles = self.ImgFiles[:nrows]

    def __len__(self):
        return len(self.ImgFiles)

    def __getitem__(self, idx):
        PathToFile = os.path.join(self.datadir, self.ImgFiles[idx])
        im = misc.imread(PathToFile)
        if len(im.shape) > 2:
            im = im[:, :, 0]
        #Add color chanel
        im = im[:,:,None]
        # Tranform
        if self.transform:
            im = self.transform(im)
        return im,PathToFile

class XrayDataset(Dataset):

    def __init__(self, datadir,DF, transform=None, nrows=-1):

        self.datadir = datadir
        self.transform = transform
        self.ImgFiles = list(DF["name"].values)
        #print(nrows)
        if nrows > 0:
            self.ImgFiles = self.ImgFiles[:nrows]

    def __len__(self):
        return len(self.ImgFiles)

    def __getitem__(self, idx):
        PathToFile = [os.path.join(self.datadir,"images/", self.ImgFiles[idx])]
        im = misc.imread(os.path.join(self.datadir,"images/", self.ImgFiles[idx]))
        if len(im.shape) > 2:
            im = im[:, :, 0]
        #Add color chanel
        im = im[:,:,None]
        # Tranform
        if self.transform:
            im = self.transform(im)

        return im,PathToFile
        
        
class Iterator:
    """
    iterator over dataloader which automatically resets when all samples have been seen
    """
    def __init__(self, dataloader):
        self.dataloader = dataloader
        self.cpt = 0
        self.len = len(self.dataloader)
        self.iterator = iter(self.dataloader)

    def next(self):
        if self.cpt == self.len:
            self.cpt = 0
            self.iterator = iter(self.dataloader)
        self.cpt += 1
        return self.iterator.next()










    
    


