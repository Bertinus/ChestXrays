from torch.utils.data import Dataset
import os
import numpy as np
import pandas as pd
from scipy import misc


def one_hot_encoding(pathology):
    """

    :param pathology: String, name of the pathology
    :return: function one hot that returns 1 if the row is labeled with pathology
    """

    def one_hot(row):
        if pathology in row['Finding Labels']:
            return 1
        return 0

    return one_hot


class XrayDataset(Dataset):

    def __init__(self, datadir, csvpath):

        self.datadir = datadir
        self.csvpath = csvpath
        self.pathologies = ["Atelectasis", "Consolidation", "Infiltration",
                            "Pneumothorax", "Edema", "Emphysema", "Fibrosis", "Effusion", "Pneumonia",
                            "Pleural_thickening", "Cardiomegaly", "Nodule", "Mass", "Hernia"]

        # Load data
        self.Data = pd.read_csv(csvpath)

        # Add one hot encodings
        for pathology in self.pathologies:
            self.Data[pathology] = self.Data.apply(one_hot_encoding(pathology), axis=1)

    def __len__(self):
        return len(self.Data)

    def __getitem__(self, idx):
        return misc.imread(os.path.join(self.datadir, self.Data['Image Index'][idx]))


if __name__=='__main__':

    ####################################################################################################################
    # %% Test
    ####################################################################################################################

    csvpath = "/home/user1/Documents/Data/ChestXray/Data_Entry_2017.csv"
    datadir = "/home/user1/Documents/Data/ChestXray/images_001/images"
    pathologies = ["Atelectasis", "Consolidation", "Infiltration",
                   "Pneumothorax", "Edema", "Emphysema", "Fibrosis", "Effusion", "Pneumonia", "Pleural_thickening",
                   "Cardiomegaly", "Nodule", "Mass", "Hernia"]

    Data = pd.read_csv(csvpath)

    for pathology in pathologies:
        Data[pathology] = Data.apply(one_hot_encoding(pathology), axis=1)

    image = misc.imread(os.path.join(datadir, Data['Image Index'][0]))

    misc.imshow(image)
