from dataset import MyDataLoader
import torch
from model import myDenseNet, addDropout
import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import imresize


def transparent_cmap(cmap, n=255):
    """
    Copy colormap and set alpha values
    """
    mycmap = cmap
    mycmap._init()
    mycmap._lut[:, -1] = np.linspace(0, 0.8, n+4)
    return mycmap


if __name__ == "__main__":

    pathologies = ["Atelectasis", "Consolidation", "Infiltration",
                   "Pneumothorax", "Edema", "Emphysema", "Fibrosis", "Effusion", "Pneumonia",
                   "Pleural_Thickening", "Cardiomegaly", "Nodule", "Mass", "Hernia"]

    inputsize = [224, 224]  # Image Size fed to the network
    batch_size = 1

    # Local
    datadir = "/home/user1/Documents/Data/ChestXray/images"
    val_csvpath = "/home/user1/Documents/Data/ChestXray/DataVal.csv"
    saved_model_path = "/home/user1/PycharmProjects/ChestXrays/Models/model_178800.pth"
    saveplotdir = "/home/user1/PycharmProjects/ChestXrays/Plots/model_test"

    """
    # Server
    datadir = "/data/lisa/data/ChestXray-NIHCC-2/images"
    val_csvpath = "/u/bertinpa/Documents/ChestXrays/Data/DataVal.csv"
    saved_model_path = "/data/milatmp1/bertinpa/Logs/model_1/model_178800.pth"
    saveplotdir = "/u/bertinpa/Documents/ChestXrays/Plots/model_178800"
    """

    # Model
    if torch.cuda.is_available():
        densenet = myDenseNet().cuda()
        densenet = addDropout(densenet, p=0)
        densenet.load_state_dict(torch.load(saved_model_path))
    else:
        densenet = myDenseNet()
        densenet = addDropout(densenet, p=0)
        densenet.load_state_dict(torch.load(saved_model_path, map_location='cpu'))

    # Dataloader
    val_dataloader = MyDataLoader(datadir, val_csvpath, inputsize, batch_size=batch_size, drop_last=True, flip=False)

    for data, label in val_dataloader:

        if torch.cuda.is_available():
            data = data.cuda()
            label = label.cuda()

        activation = densenet(data)[-2][0].transpose(0, 2).detach()  # Activation before last FC layer
        classif_weight = densenet.classifier[0][0].weight.detach()  # Weight of the last FC layer

        # List of heatmaps to be plotted
        heatmaps = [torch.sum(classif_weight[i] * activation, dim=2).numpy().clip(min=0)
                    for i in range(len(classif_weight))]

        data = data.numpy()[0, 0]
        heatmaps = [imresize(x, data.shape)*np.max(x)/255 for x in heatmaps]

        # Use base cmap to create transparent
        mycmap = transparent_cmap(plt.cm.Reds)
        y, x = np.mgrid[0:224, 0:224]

        fig, ax = plt.subplots(1, 1)
        plt.imshow(data, cmap="gray")
        plt.axis('off')
        plt.title("Original image")
        plt.show()

        # For a given image, plot 14 heatmaps corresponding to the 14 deseases 
        for i in range(14):
            fig, ax = plt.subplots(1, 1)
            plt.imshow(data, cmap="gray")
            if not (heatmaps[i] == 0).all():
                cb = ax.contourf(x, y, heatmaps[i], 8, cmap=mycmap, vmin=0, vmax=8)
                # fig.colorbar(cb)
            plt.title(pathologies[i] + " (ground truth : " + str(label[0, i].numpy())[0] + ")")
            plt.axis('off')
            plt.show()

        quit()