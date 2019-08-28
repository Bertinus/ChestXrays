import os
import pandas as pd
from scipy import misc
from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt


class PadchestDataset(Dataset):

    def __init__(self, datadir, csvpath, transform=None, nrows=None):

        self.datadir = datadir
        self.transform = transform
        # Load data
        Data = pd.read_csv(csvpath, nrows=nrows, low_memory=False)
        Data = Data[Data.Projection == "PA"]
        self.Data = Data.reset_index(drop=True)

        self.pathologies = ["atelectasis", "consolidation", "infiltrates",
                            "pneumothorax", "edema", "emphysema", "fibrosis", "effusion", "pneumonia",
                            "pleural thickening", "cardiomegaly", "nodule", "mass", "hernia"]

        for patho in self.pathologies:
            self.Data[patho] = self.Data.Labels.str.contains(patho).fillna(False)

    def __len__(self):
        return len(self.Data)

    def __getitem__(self, idx):

        try:
            im = misc.imread(os.path.join(self.datadir, self.Data['ImageID'][idx]))
        except OSError:
            print("Missing image, reading next one instead")
            im = misc.imread(os.path.join(self.datadir, self.Data['ImageID'][idx+1]))

        im = im.astype(float)
        im /= np.max(im)
        im *= 255
        im = im.astype(np.int32)

        # Tranform
        if self.transform:
            im = self.transform(im[:, :, None])

        return im, self.Data[self.pathologies].loc[idx].values.astype(np.float32)


def PadChestDataLoader(datadir, csvpath, inputsize, batch_size=16, nrows=None, drop_last=False, data_transforms=None,
                       shuffle=True):
    # Transformations
    if data_transforms is None:
        data_transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(inputsize),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1))
        ])

    # Initialize dataloader
    dataset = PadchestDataset(datadir, csvpath, transform=data_transforms, nrows=nrows)
    dataloader = DataLoader(dataset, shuffle=shuffle, batch_size=batch_size, drop_last=drop_last)

    return dataloader


class XrayDataset(Dataset):

    def __init__(self, datadir, csvpath, transform=None, nrows=None):

        self.datadir = datadir
        self.transform = transform
        self.pathologies = ["Atelectasis", "Consolidation", "Infiltration",
                            "Pneumothorax", "Edema", "Emphysema", "Fibrosis", "Effusion", "Pneumonia",
                            "Pleural_Thickening", "Cardiomegaly", "Nodule", "Mass", "Hernia"]

        # Load data
        self.Data = pd.read_csv(csvpath, nrows=nrows)

    def __len__(self):
        return len(self.Data)

    def __getitem__(self, idx):
        im = misc.imread(os.path.join(self.datadir, self.Data['Image Index'][idx]))
        # For the ChestXRay dataset, range is [0, 255]

        # Check that images are 2D arrays
        if len(im.shape) > 2:
            im = im[:, :, 0]
        if len(im.shape) < 2:
            print("error, dimension lower than 2 for image", self.Data['Image Index'][idx])

        # Add color channel
        im = im[:, :, None]

        # Tranform
        if self.transform:
            im = self.transform(im)

        return im, self.Data[self.pathologies].loc[idx].values.astype(np.float32), self.Data['Image Index'][idx]


def MyDataLoader(datadir, csvpath, inputsize, batch_size=16, nrows=None, drop_last=False, data_transforms=None,
                 shuffle=True):
    # Transformations
    if data_transforms is None:
        data_transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(inputsize),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1))
        ])

    # Initialize dataloader
    dataset = XrayDataset(datadir, csvpath, transform=data_transforms, nrows=nrows)
    dataloader = DataLoader(dataset, shuffle=shuffle, batch_size=batch_size, drop_last=drop_last)

    return dataloader


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


if __name__ == '__main__':

    ####################################################################################################################
    # Test PadChest
    ####################################################################################################################

    datadir = "/home/user1/Documents/Data/PADCHEST/sample"
    csvpath = "/home/user1/Documents/Data/PADCHEST/chest_x_ray_images_labels_sample.csv"
    # csvpath = "/home/user1/Documents/Data/PADCHEST/PADCHEST_chest_x_ray_images_labels_160K_01.02.19.csv"

    dataloader = PadChestDataLoader(datadir=datadir, csvpath=csvpath, inputsize=[224, 224], batch_size=8)

    for data in dataloader:
        for i in range(3):
            print(data)
            plt.imshow(np.rollaxis(data[0][i].numpy(), 0, 3))
            plt.show()

    ####################################################################################################################
    # Test
    ####################################################################################################################

    datadir = "/home/user1/Documents/Data/ChestXray/images"
    csvpath = "/home/user1/Documents/Data/ChestXray/DataTrain.csv"
    inputsize = [224, 224]

    # Transformations
    data_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomAffine(15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        # transforms.RandomVerticalFlip(),
        transforms.Resize(inputsize),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1))
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Initialize dataloader
    dataset = XrayDataset(datadir, csvpath, transform=data_transforms, nrows=100)
    dataloader = DataLoader(dataset, shuffle=True, batch_size=16)

    print(len(dataset))

    cpt = 0

    for dataiter in dataloader:
        data, label, idx = dataiter

        cpt += 1
        if cpt > 1:
            break

        print("data", data.size())
        print("label", label.size())
        print("pixel range", data.min(), data.max())
        print("pixel mean", data[:, 0].mean(), data[:, 1].mean(), data[:, 2].mean())
        print("pixel std", data[:, 0].std(), data[:, 1].std(), data[:, 2].std())

        for i in range(3):
            plt.imshow(np.rollaxis(data[i].numpy(), 0, 3))
            plt.show()
