import os
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import ShuffleSplit
import matplotlib.pyplot as plt
import pandas as pd
from torchvision import transforms
from model import myDenseNet, addDropout
import imageio
from tqdm import tqdm


class XrayDataset(Dataset):

    def __init__(self, datadir, csvpath, transform=None, nrows=None):

        self.datadir = datadir
        self.transform = transform
        self.pathologies = ["Atelectasis", "Consolidation", "Infiltration",
                            "Pneumothorax", "Edema", "Emphysema", "Fibrosis", "Effusion", "Pneumonia",
                            "Pleural_Thickening", "Cardiomegaly", "Nodule", "Mass", "Hernia"]

        # Load data
        self.Data = pd.read_csv(csvpath, nrows=nrows, sep=',')

    def __len__(self):
        return len(self.Data)

    def __getitem__(self, idx):
        im = imageio.imread(os.path.join(self.datadir, self.Data['Image Index'][idx]))

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

        return im, self.Data[self.pathologies].loc[idx].values.astype(np.float32), idx


def MyDataLoader(datadir, csvpath, inputsize, batch_size=16, nrows=None, drop_last=False, flip=True):
    # Transformations
    if flip:
        data_transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(),
            # transforms.RandomVerticalFlip(),
            transforms.Resize(inputsize),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1))
        ])
    else:
        data_transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(inputsize),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1))
        ])

    # Initialize dataloader
    dataset = XrayDataset(datadir, csvpath, transform=data_transforms, nrows=nrows)
    dataloader = DataLoader(dataset, shuffle=True, batch_size=batch_size, drop_last=drop_last)

    return dataloader


if __name__ == "__main__":

    ####################################################################################################################
    # Parameters
    ####################################################################################################################

    pathologies = ["Atelectasis", "Consolidation", "Infiltration",
                   "Pneumothorax", "Edema", "Emphysema", "Fibrosis", "Effusion", "Pneumonia",
                   "Pleural_Thickening", "Cardiomegaly", "Nodule", "Mass", "Hernia"]

    original_paper_results = ["0.8094", "0.7901", "0.7345",
                              "0.8887", "0.8878", "0.9371", "0.8047", "0.8638", "0.7680",
                              "0.8062", "0.9248", "0.7802", "0.8676", "0.9164"]

    # Local
    """
    datadir = "/home/user1/Documents/Data/ChestXray/images"
    # val_csvpath = "/home/user1/Documents/Data/ChestXray/DataVal.csv"
    test_csvpath = "/home/user1/PycharmProjects/ChestXrays/Old/arnowengtest.csv"
    saved_model_path = "/home/user1/PycharmProjects/ChestXrays/Models/model.pth.tar"
    saveplotdir = "/home/user1/PycharmProjects/ChestXrays/Plots/model_test"

    """
    # Server
    datadir = "/network/data1/ChestXray-NIHCC-2/images"
    test_csvpath = "/network/home/bertinpa/Documents/ChestXrays/Data/DataVal.csv"
    saved_model_path = "Models/model_72800.pth"  # "Models/model_178800.pth"
    saveplotdir = "/network/home/bertinpa/Documents/ChestXrays/Plots/test"

    inputsize = [224, 224]  # Image Size fed to the network
    batch_size = 16
    n_batch = -1  # Number of batches used to compute the AUC, -1 for all validation set
    n_splits = 10  # Number of randomized splits to compute standard deviations
    split = ShuffleSplit(n_splits=n_splits, test_size=0.5, random_state=0)

    ####################################################################################################################
    # Compute predictions
    ####################################################################################################################

    val_dataloader = MyDataLoader(datadir, test_csvpath, inputsize, batch_size=batch_size, drop_last=True, flip=False)
    all_data = pd.read_csv(test_csvpath)

    if n_batch == -1:
        n_batch = len(val_dataloader)

    print("Total number of batches", len(val_dataloader))

    # Initialize result arrays
    all_outputs = np.zeros((n_batch * batch_size, 14))
    all_labels = np.zeros((n_batch * batch_size, 14))
    all_idx = np.zeros((n_batch * batch_size, 1))

    # Model
    if torch.cuda.is_available():
        densenet = myDenseNet().cuda()
        densenet = addDropout(densenet, p=0)
        densenet.load_state_dict(torch.load(saved_model_path))
        # densenet = DenseNet121(14).cuda()
        # densenet = addDropout(densenet, p=0)
        # densenet.load_state_dict(torch.load(saved_model_path))
    else:
        densenet = myDenseNet()
        densenet = addDropout(densenet, p=0)
        densenet.load_state_dict(torch.load(saved_model_path, map_location='cpu'))
        # densenet = DenseNet121(14)
        # densenet = addDropout(densenet, p=0)
        # densenet.load_state_dict(load_dictionary(saved_model_path, map_location='cpu'))

    cpt = 0
    densenet.eval()

    for data, label, idx in tqdm(val_dataloader):

        if torch.cuda.is_available():
            data = data.cuda()
            label = label.cuda()

        output = densenet(data)[-1]

        if torch.cuda.is_available():
            all_labels[cpt * batch_size: (cpt + 1) * batch_size] = label.detach().cpu().numpy()
            all_outputs[cpt * batch_size: (cpt + 1) * batch_size] = output.detach().cpu().numpy()
            all_idx[cpt * batch_size: (cpt + 1) * batch_size] = idx.detach().cpu().numpy()[:, None]
        else:
            all_labels[cpt * batch_size: (cpt + 1) * batch_size] = label.detach().numpy()
            all_outputs[cpt * batch_size: (cpt + 1) * batch_size] = output.detach().numpy()
            all_idx[cpt * batch_size: (cpt + 1) * batch_size] = idx.detach().numpy()[:, None]

        cpt += 1
        if cpt == n_batch:
            break

    # Save predictions as a csv
    all_names = np.array([all_data['Image Index'][idx].values for idx in all_idx])
    csv_array = pd.DataFrame(np.concatenate((all_names, all_outputs, all_labels), axis=1))
    column_names = ["name"] + ["prediction_"+str(i) for i in range(14)] + ["label_"+str(i) for i in range(14)]
    csv_array.to_csv("model_predictions.csv", header=column_names, index=False)

    np.save("all_outputstest", all_outputs)
    np.save("all_labelstest", all_labels)

    ####################################################################################################################
    # Compute AUC and operating points
    ####################################################################################################################

    all_outputs = np.load("all_outputsval.npy")
    all_labels = np.load("all_labelsval.npy")

    print(all_labels.shape)

    print("# Results\n\n| desease | original paper  |  git model  |\n|---|---|---|")

    plt.figure(0, figsize=(17, 5))

    for i in range(14):
        if (all_labels[:, i] == 0).all():
            print("|", pathologies[i], "|", original_paper_results[i], "|", "ERR |")
        else:
            # Compute AUC and STD with randomized splits
            split_auc = [roc_auc_score(all_labels[split_index, i], all_outputs[split_index, i])
                         for split_index, _ in split.split(all_outputs) if not (all_labels[split_index, i] == 0).all()]

            auc = np.mean(split_auc)
            std = np.std(split_auc)

            print("|", pathologies[i], "|", original_paper_results[i], "|",
                  str(auc)[:6], "+-", str(std)[:6], "|")

            # Save ROC curve
            fpr, tpr, thres = roc_curve(all_labels[:, i], all_outputs[:, i])

            # Compute operating point
            pente = tpr - fpr

            opt_thres = thres[np.argmax(pente)]
            opt_fpr = fpr[np.argmax(pente)]
            opt_tpr = tpr[np.argmax(pente)]

            print(pathologies[i], "opt thresh", opt_thres, "opt fpr", opt_fpr, "opt tpr", opt_tpr)

            if i > 6:
                xplot = 1
                yplot = i-7
            else:
                xplot = 0
                yplot = i

            auc = roc_auc_score(all_labels[:, i], all_outputs[:, i])

            ax = plt.subplot2grid((2, 7), (xplot, yplot))

            ax.plot(fpr, tpr)
            ax.plot(np.arange(0, 1.1, 0.1), np.arange(0, 1.1, 0.1))
            ax.scatter(opt_fpr, opt_tpr, marker="*", c="r", s=100)
            ax.set_title(pathologies[i] + " ROC")
            ax.tick_params(top='off', bottom='off', left='off', right='off')
            ax.set_yticks([0, 0.5, 1])
            ax.set_xlabel("FPR")
            ax.set_ylabel("TPR")
            ax.text(0.51, 0.2, "AUC=" + str(auc)[:4])
    plt.show()
