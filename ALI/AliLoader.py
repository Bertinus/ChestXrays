from torch.utils.data import Dataset
import glob
import imageio
import os


class XrayDataset(Dataset):

    def __init__(self, datadir, transform=None, nrows=None):

        self.datadir = datadir
        self.transform = transform
        self.ImgFiles = [f.split('/')[-1] for f in glob.glob(datadir+"*.png")]

    def __len__(self):
        return len(self.ImgFiles)

    def __getitem__(self, idx):
        im = misc.imread(os.path.join(self.datadir, self.ImgFiles[idx]))
        
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
