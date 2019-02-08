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

#Function that only load pre format training set (list From Kaggle)
def LoadTrainTestSet(datadir,isize,rseed=13,subset="Training",N=-1,verbose=0,split="new"):
    #PreProcess folder
    
    PreProDir = datadir+"PreProcess/Size"+str(isize)
    ImagesInfoDF = pd.read_csv(PreProDir+"/AllImagesInfo.csv")
    if verbose == 1:
        print("Getting split")
    #Get Training set
    TrainingSet = pd.read_table(datadir+"/train_val_list.txt",names=["name"]).sample( random_state=rseed,frac=1.0)
    if subset != "Training":
        TrainingSet = pd.read_table(datadir+"/test_list.txt",names=["name"]).sample( random_state=rseed,frac=1.0)
    if split != "new":
        #Get Training set
        TrainingSet = pd.read_csv(PreProDir+"/TrainImagesInfo.csv").sample( random_state=rseed,frac=1.0)
        if subset != "Training":
            TrainingSet = pd.read_csv(PreProDir+"/TestImagesInfo.csv").sample( random_state=rseed,frac=1.0)
    #Find overlap between list and preformatted
    TrainSet = np.intersect1d(TrainingSet["name"],ImagesInfoDF["name"])
    if N > 0:
        TrainSet = TrainSet[:N]
    #Load and dataloader
    
    train_dataset = XrayDatasetTensor(
        PreProDir+"/Tensor"+str(isize)+".pt",PreProDir+"/AllImagesInfo.csv",list(TrainSet),verbose=verbose)
    #Return dataloader
    return(train_dataset)

def LoadMNIST(datadir,isize):
    #Load MNIST
    MNIST_transform = transforms.Compose([transforms.Resize(isize),transforms.ToTensor()])
    MNIST_set = dset.MNIST(root=datadir, train=True, transform=MNIST_transform, download=True)
    
    return(MNIST_set)


#Load Img list and transform to tensor
class ImgFileLoader(Dataset):

    def __init__(self, FilesList,isize,label=["NA"],shuffle=False,ExtraTransf = None,Mean=False,verbose=0):

        self.ImgFiles = FilesList
        
        ImgTensor = torch.tensor([])
        #Transform each images to tensor
        c = 0
        for PathToFile in self.ImgFiles:
            im = misc.imread(PathToFile)
            if len(im.shape) > 2:
                im = im[:, :, 0]
            #Add color chanel
            im = im[:,:,None]
            
            #Make image square! (pad with black pixel)
            padding = 0
            if im.shape[0] > im.shape[1]:
                padding = (int((im.shape[0]-im.shape[1])/2),0)
            else:
                padding = (0,int((im.shape[1]-im.shape[0])/2))
            #Build transformatin
            TransfArr = [transforms.ToPILImage(),transforms.Pad(padding,fill=0)]
            
            if ExtraTransf:
                for t in ExtraTransf:
                    TransfArr.append(t)
            
            #Resize and transform to Tensor
            TransfArr.append(transforms.Resize([isize,isize]))
            TransfArr.append(transforms.ToTensor())
            data_transforms = transforms.Compose(TransfArr)
            # Tranform    
            im = data_transforms(im)
            im = im.reshape(1,1,im.shape[2],im.shape[2])
            
            
            if shuffle == True:
                # With view
                idx = torch.randperm(im.nelement())
                im = im.view(-1)[idx].view(im.size())
            if Mean == True:
                im = im - im + (float(c)/float(len(self.ImgFiles)))
            c += 1
            if verbose == 1:
                if c % 1000 == 0:
                    print(c,len(self.ImgFiles))
            ImgTensor = torch.cat((ImgTensor, im), 0)    
            
        self.ImgTensor = ImgTensor
        if label[0] == "NA":
            label = ["NA"]*len(FilesList)
        self.types = label
        
        

    def __len__(self):
        return len(self.ImgFiles)

    def __getitem__(self, idx):
        PathToFile = self.ImgFiles[idx]
        types = self.types[idx]
        im = self.ImgTensor[idx]
        
        
        return im,[PathToFile,types]
def LoadCIFAR(datadir,isize):
    #Load MNIST
    CIFAR_transform = transforms.Compose([transforms.Grayscale(num_output_channels=1),transforms.Resize(isize),transforms.ToTensor()])
    CIFAR_set = dset.CIFAR100(root=datadir, train=True, transform=CIFAR_transform, download=True)
    
    return(CIFAR_set)

def LoadMURA(datadir,isize,N=-1,rseed = -1,verbose=0):
    
    #Build file list
    if not os.path.isfile(datadir+"FilesList.csv"):
        tFilesList =sorted(glob.glob(datadir+"/*/*/*/*/*.png"))
        pd.DataFrame(tFilesList).to_csv(datadir+"FilesList.csv")
        
    #Load File list
    FilesList = pd.read_csv(datadir+"FilesList.csv")['0'].values
    if verbose == 1:
        print("MURA File List = %d" % (len(FilesList)))
    if rseed >= 0:
        np.random.seed(rseed)
        FilesList = np.random.permutation(FilesList)
    
    #File to keep
    if N > 0:
        FilesList = FilesList[:N]
        
    label = []
    #Get Label
    for f in FilesList:
        pos = f.split("/")[-2].split("_")[-1]
        reg = f.split("/")[-4].split("_")[-1]
        label.append("_".join([pos,reg]))
    
    MURAset = ImgFileLoader(FilesList,isize,label=label,verbose=verbose)
    return(MURAset)


def LoadPneunomia(datadir,isize,N=-1,rseed = -1):
    
    
    #Build file list
    if not os.path.isfile(datadir+"FilesList.csv"):
        tFilesList =sorted(glob.glob(datadir+"/*/*/*.*"))
        pd.DataFrame(tFilesList).to_csv(datadir+"FilesList.csv")
    
    FilesList = pd.read_csv(datadir+"FilesList.csv")['0'].values
    if rseed >= 0:
        np.random.seed(rseed)
        FilesList = np.random.permutation(FilesList)
    
    #File to keep
    if N > 0:
        FilesList = FilesList[:N]
        
    label = []
    #Get Label
    for f in FilesList:
        
        label.append(f.split("/")[-2])
    Pneuno = ImgFileLoader(FilesList,isize,label=label)
    return(Pneuno)


#Load modified chest x-ray (flip, permuted and random)
def LoadModChest(datadir,isize,N=-1,rseed=13,subset="Testing",split = "new",check_file = False):
    #Get Training set
    TrainingSet = pd.read_table(datadir+"/train_val_list.txt",names=["name"]).sample( random_state=rseed,frac=1.0)
    if subset != "Training":
        TrainingSet = pd.read_table(datadir+"/test_list.txt",names=["name"]).sample( random_state=rseed,frac=1.0)
        
    if split != "new":
        #Get Training set
        TrainingSet = pd.read_csv(datadir+"/TrainImagesInfo.csv").sample( random_state=rseed,frac=1.0)
        if subset != "Training":
            TrainingSet = pd.read_csv(datadir+"/TestImagesInfo.csv").sample( random_state=rseed,frac=1.0)
    FilesList = []
    if check_file == True:
        print("Globing")
        GlobList = glob.glob(datadir+"/images/*.png")
        for f in GlobList:
            if f.split("/")[-1] in list(TrainingSet["name"].values):
                FilesList.append(f)
        print(len(GlobList),len(FilesList))
    else:
        for fs in list(TrainingSet["name"].values):
             f = datadir+"/images/" + fs 
             FilesList.append(f)
            
    if N > 0:
        FilesList = np.random.permutation(FilesList)
        FilesList = FilesList[:N]
    
    print("Loading Hflip")
    Hflip = ImgFileLoader(FilesList,isize,label=["NA"]*len(FilesList),ExtraTransf=[transforms.RandomHorizontalFlip(p=1.0)])
    print("Loading Vflip")
    Vflip = ImgFileLoader(FilesList,isize,label=["NA"]*len(FilesList),ExtraTransf=[transforms.RandomVerticalFlip(p=1.0)])
    
    print("Loading Shuffle")
    Shuffle = ImgFileLoader(FilesList,isize,label=["NA"]*len(FilesList),shuffle=True)
    print("Loading mean")
    Mean = ImgFileLoader(FilesList,isize,label=["NA"]*len(FilesList),Mean=True)
    
    RandomTransf = transforms.RandomChoice([
        transforms.RandomAffine(degrees=[-90,-15],translate=(0.1,0.1),scale=(1,1.2)),
        transforms.RandomAffine(degrees=[15,90],translate=(0.1,0.1),scale=(1,1.2))
    ])
    print("Loading Random transformation")
    Random = ImgFileLoader(FilesList,isize,label=["NA"]*len(FilesList),
                           ExtraTransf=[RandomTransf])
    
    return(Hflip,Vflip,Shuffle,Random,Mean)

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






class XrayDataset(Dataset):

    def __init__(self, datadir,DF, transform=None, nrows=-1,shuffle=False):

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
            
            if shuffle == True:

                # With view
                idx = torch.randperm(im.nelement())
                im = im.view(-1)[idx].view(im.size())
            ImgTensor = torch.cat((ImgTensor, im), 0)
        self.ImgTensor = ImgTensor

         
    def __len__(self):
        return len(self.ImgFiles)

    def __getitem__(self, idx):
        PathToFile = os.path.join(self.datadir,"images/", self.ImgFiles[idx])
        im = self.ImgTensor[idx]
        return im,[PathToFile,PathToFile]



class XrayDatasetTensor(Dataset):

    def __init__(self, TensorName,FullDF,Names,verbose=0):
        if verbose == 1:print("Loading Tensor")
            
        self.ImgTensor = torch.load(TensorName)
        if verbose == 1:print("Tensor loaded")
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
