from dataset import MyDataLoader
import torch
import os
from model import myDenseNet
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt

if __name__ == "__main__":

    ####################################################################################################################
    # Parameters
    ####################################################################################################################

    pathologies = ["Atelectasis", "Consolidation", "Infiltration",
                   "Pneumothorax", "Edema", "Emphysema", "Fibrosis", "Effusion", "Pneumonia",
                   "Pleural_thickening", "Cardiomegaly", "Nodule", "Mass", "Hernia"]
    """
    # Local
    datadir = "/home/user1/Documents/Data/ChestXray/images"
    val_csvpath = "/home/user1/Documents/Data/ChestXray/DataVal.csv"
    saved_model_path = "/home/user1/PycharmProjects/ChestXrays/Models/model_39000.pth"
    saveplotdir = "/home/user1/PycharmProjects/ChestXrays/Plots/model_39000"

    """
    # Server
    datadir = "/data/lisa/data/ChestXray-NIHCC-2/images"
    val_csvpath = "/u/bertinpa/Documents/ChestXrays/Data/DataVal.csv"
    saved_model_path = "/u/bertinpa/Documents/ChestXrays/Logs/model_1/model_39000.pth"
    saveplotdir = "/u/bertinpa/Documents/ChestXrays/Plots/model_39000"

    inputsize = [224, 224]  # Image Size fed to the network
    batch_size = 16
    n_batch = -1  # Number of batches used to compute the AUC, -1 for all validation set

    ####################################################################################################################
    # Compute predictions
    ####################################################################################################################

    val_dataloader = MyDataLoader(datadir, val_csvpath, inputsize, batch_size=batch_size)

    if n_batch == -1:
        n_batch = len(val_dataloader)

    # Initialize result arrays
    all_outputs = np.zeros((n_batch * batch_size, 14))
    all_labels = np.zeros((n_batch * batch_size, 14))

    # Model
    if torch.cuda.is_available():
        densenet = myDenseNet().cuda()
        densenet.load_state_dict(torch.load(saved_model_path))
    else:
        densenet = myDenseNet()
        densenet.load_state_dict(torch.load(saved_model_path, map_location='cpu'))

    cpt = 0
    densenet.eval()

    for data, label in val_dataloader:

        if torch.cuda.is_available():
            data = data.cuda()
            label = label.cuda()

        output = densenet(data)[-1]

        if torch.cuda.is_available():
            all_labels[cpt*batch_size: (cpt+1)*batch_size] = label.detach().cpu().numpy()
            all_outputs[cpt*batch_size: (cpt + 1) * batch_size] = output.detach().cpu().numpy()
        else:
            all_labels[cpt*batch_size: (cpt+1)*batch_size] = label.detach().numpy()
            all_outputs[cpt*batch_size: (cpt + 1) * batch_size] = output.detach().numpy()

        cpt += 1
        if cpt == n_batch:
            break

    ####################################################################################################################
    # Compute AUC
    ####################################################################################################################

    for i in range(14):
        if (all_labels[:, i] == 0).all():
            print(pathologies[i], "ERR")
        else:
            auc = roc_auc_score(all_labels[:, i], all_outputs[:, i])
            print(pathologies[i], auc)
            fpr, tpr, thres = roc_curve(all_labels[:, i], all_outputs[:, i])
            plt.subplot(121)
            plt.plot(fpr, tpr)
            plt.text(0.6, 0.2, "AUC="+str(auc)[:4])
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title(pathologies[i] + " ROC")
            plt.subplot(122)
            plt.hist(all_outputs[:, i], range=(0, 1), bins=50)
            plt.title(pathologies[i] + " Prediction histo")
            plt.savefig(os.path.join(saveplotdir, str(pathologies[i] + '.png')))
            plt.clf()
