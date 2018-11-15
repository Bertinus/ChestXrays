from torch.utils.data import Dataset
import glob
import imageio
import os
from scipy import misc
import pandas as pd



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
        im = misc.imread(os.path.join(self.datadir, self.ImgFiles[idx]))
        if len(im.shape) > 2:
            im = im[:, :, 0]
        #Add color chanel
        im = im[:,:,None]
        # Tranform
        if self.transform:
            im = self.transform(im)
        return im

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
        im = misc.imread(os.path.join(self.datadir,"images/", self.ImgFiles[idx]))
        if len(im.shape) > 2:
            im = im[:, :, 0]
        #Add color chanel
        im = im[:,:,None]
        # Tranform
        if self.transform:
            im = self.transform(im)
        return im
        
        
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










    
    


