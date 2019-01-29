#%%
import pandas as pd
from sklearn.model_selection import train_test_split
import os
import numpy as np


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


#%%
if __name__ == "__main__":

    ####################################################################################################################
    # %% Build Train Test Val Datasets
    ####################################################################################################################

    # Local
    testListPath = "/home/user1/Documents/Data/ChestXray/Utils/test_list.txt"
    csvpath = "/home/user1/Documents/Data/ChestXray/Data_Entry_2017.csv"
    savepath = "/home/user1/Documents/Data/ChestXray"

    """
    # Server
    testListPath = "/u/bertinpa/Documents/ChestXrays/Data/test_list.txt"
    csvpath = "/data/lisa/data/ChestXray-NIHCC-2/Data_Entry_2017.csv"
    savepath = "/u/bertinpa/Documents/ChestXrays/Data"
    """
    # Load data
    Data = pd.read_csv(csvpath)
    testList = pd.read_csv(testListPath, header=None)[0].tolist()

    # Add one hot encodings
    pathologies = ["Atelectasis", "Consolidation", "Infiltration",
                        "Pneumothorax", "Edema", "Emphysema", "Fibrosis", "Effusion", "Pneumonia",
                        "Pleural_Thickening", "Cardiomegaly", "Nodule", "Mass", "Hernia"]
    for pathology in pathologies:
        Data[pathology] = Data.apply(one_hot_encoding(pathology), axis=1)

    # Split Test vs TrainVal using the same splitting as the authors of the Dataset
    DataTest = Data.loc[Data["Image Index"].isin(testList)]
    DataTrainVal = Data.loc[~Data["Image Index"].isin(testList)]
    DataTrainVal.reset_index(inplace=True)

    # Split Train vs Validation on patient level
    patients = DataTrainVal['Patient ID'].unique().tolist()
    patients_train, patients_val = train_test_split(patients, test_size=0.11, random_state=14)

    DataTrain = DataTrainVal.loc[DataTrainVal['Patient ID'].isin(patients_train)]
    DataVal = DataTrainVal.loc[DataTrainVal['Patient ID'].isin(patients_val)]
    Train_idx_list = DataTrain.index.tolist()
    Val_idx_list = DataVal.index.tolist()

    print("validation proportion :", DataVal.shape[0] / DataTrainVal.shape[0])

    ####################################################################################################################
    # %% Save as csv
    ####################################################################################################################

    # Save csv corresponding to train, test and validation
    DataTrainVal.to_csv(os.path.join(savepath, "DataTrainVal.csv"), index=False)
    DataTrain.to_csv(os.path.join(savepath, "DataTrain.csv"), index=False)
    DataVal.to_csv(os.path.join(savepath, "DataVal.csv"), index=False)
    np.save(os.path.join(savepath, "Train_Idx_List"), Train_idx_list)
    np.save(os.path.join(savepath, "Val_Idx_List"), Val_idx_list)
    DataTest.to_csv(os.path.join(savepath, "DataTest.csv"), index=False)



