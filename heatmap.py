from dataset import MyDataLoader, Iterator
import torch
from model import averageCrossEntropy, DenseNet121, load_dictionary
import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import imresize


def transparent_cmap(cmap, n=255):
    """
    Copy colormap and set alpha values
    """
    mycmap = cmap
    mycmap._init()
    mycmap._lut[:, -1] = np.linspace(0, 0.8, n + 4)
    return mycmap


def get_heatmap(classif_weight, activation, data):
    """
    :param classif_weight:
    :param activation:
    :param data:
    :return: list of heatmaps to be plotted corresponding to the different deseases
    """
    heatmaps = [torch.sum(classif_weight[i] * activation, dim=2).numpy().clip(min=0)
                for i in range(len(classif_weight))]

    heatmaps = [imresize(x, data.detach().numpy()[0, 0].shape) * np.max(x) / 255 for x in heatmaps]

    return heatmaps


def plot_heatmaps(data, label, heatmaps, pathologies):
    # Use base cmap to create transparent
    mycmap = transparent_cmap(plt.cm.Reds)
    y, x = np.mgrid[0:224, 0:224]

    # Plot original images
    fig, ax = plt.subplots(1, 1)
    plt.imshow(data.detach().numpy()[0, 0], cmap="gray")
    plt.axis('off')
    plt.title("Original image")
    plt.show()

    # For a given image, plot 14 heatmaps corresponding to the 14 deseases
    for i in range(len(pathologies)):
        fig, ax = plt.subplots(1, 1)
        plt.imshow(data.detach().numpy()[0, 0], cmap="gray")
        if not (heatmaps[i] == 0).all():
            cb = ax.contourf(x, y, heatmaps[i], 8, cmap=mycmap, vmin=0, vmax=8)
            fig.colorbar(cb)
        plt.title(pathologies[i] + " (ground truth : " + str(label[0, i].numpy())[0] + ")")
        plt.axis('off')
        plt.show()


if __name__ == "__main__":

    ####################################################################################################################
    # Parameters
    ####################################################################################################################

    inputsize = [224, 224]  # Image Size fed to the network
    batch_size = 1

    pathologies = ["Atelectasis", "Consolidation", "Infiltration",
                   "Pneumothorax", "Edema", "Emphysema", "Fibrosis", "Effusion", "Pneumonia",
                   "Pleural_Thickening", "Cardiomegaly", "Nodule", "Mass", "Hernia"]

    """
    # Local
    datadir = "/home/user1/Documents/Data/ChestXray/images"
    val_csvpath = "/home/user1/Documents/Data/ChestXray/DataVal.csv"
    saved_model_path = "Models/model.pth.tar"  # "Models/model_178800.pth"

    """
    # Server
    datadir = "/data/lisa/data/ChestXray-NIHCC-2/images"
    val_csvpath = "/u/bertinpa/Documents/ChestXrays/Data/DataVal.csv"
    saved_model_path = "/data/milatmp1/bertinpa/Logs/model_1/model.pth.tar"

    ####################################################################################################################
    # Initialization
    ####################################################################################################################

    # Model
    if torch.cuda.is_available():
        # densenet = myDenseNet().cuda()
        # densenet = addDropout(densenet, p=0)
        # densenet.load_state_dict(torch.load(saved_model_path))
        # If pretrained model from git repo
        densenet = DenseNet121(14)
        densenet.load_state_dict(load_dictionary(saved_model_path))
    else:
        # densenet = myDenseNet()
        # densenet = addDropout(densenet, p=0)
        # densenet.load_state_dict(torch.load(saved_model_path, map_location='cpu'))
        # If pretrained model from git repo
        densenet = DenseNet121(14)
        densenet.load_state_dict(load_dictionary(saved_model_path, map_location='cpu'))

    densenet.eval()

    # Dataloader
    val_dataloader = MyDataLoader(datadir, val_csvpath, inputsize, batch_size=batch_size, drop_last=True, flip=False)

    # Loss
    criterion = averageCrossEntropy

    val_iterator = Iterator(val_dataloader)  # Iterator for validation samples

    data, label, idx = val_iterator.next()

    data.requires_grad = True

    if torch.cuda.is_available():
        data = data.cuda()
        label = label.cuda()

    output = densenet(data)

    ####################################################################################################################
    # Plot heatmaps based on last layer activations
    ####################################################################################################################

    activation = output[-2][0].transpose(0, 2).detach()  # Activation before last FC layer
    classif_weight = densenet.classifier[0][0].weight.detach()  # Weight of the last FC layer
    heatmaps = get_heatmap(classif_weight, activation, data)

    plot_heatmaps(data, label, heatmaps, pathologies)

    ####################################################################################################################
    # Plot heatmaps based on gradient norm
    ####################################################################################################################

    # heatmaps = []
    #
    # for i in range(14):
    #     class_label = torch.zeros((1, 14))
    #     class_label[0, i] = 1
    #
    #     loss = criterion(output, class_label)
    #     loss.backward(retain_graph=True)
    #
    #     heatm = np.absolute(data.grad.detach().numpy()[0, 0])
    #     heatm = 10 * heatm / np.max(heatm)
    #
    #     heatmaps.append(heatm)
    #
    #     data.grad.zero_()
    #
    # plot_heatmaps(data, label, heatmaps, pathologies)
