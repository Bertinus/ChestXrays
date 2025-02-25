from dataset import MyDataLoader, PadChestDataLoader
import torch
import os
from model import myDenseNet, addDropout, DenseNet121, load_dictionary
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import ShuffleSplit
import matplotlib.pyplot as plt
from torchvision import transforms
import pandas as pd
import argparse
from tqdm import tqdm

if __name__ == "__main__":

    ####################################################################################################################
    # Parameters
    ####################################################################################################################

    # Default paths
    default_datadir = "/network/data1/academictorrents-datastore/PADCHEST_SJ/image"
    default_val_csvpath = "/network/home/bertinpa/Documents/ChestXrays/PADCHEST_chest_x_ray_images_labels_160K_" \
                          "01.02.19_cleaned.csv"
    default_saveplotdir = "/network/home/bertinpa/Documents/ChestXrays/Plots/test"
    # "/network/tmp1/bertinpa/Logs/model.pth.tar"
    # datadir = "/network/data1/ChestXray-NIHCC-2/images"
    # val_csvpath = "/network/home/bertinpa/Documents/ChestXrays/Data/DataVal.csv"

    parser = argparse.ArgumentParser()
    parser.add_argument('--datadir', default=default_datadir, type=str,
                        help='Path to images - Default is Padchest')
    parser.add_argument('--val_csvpath', default=default_val_csvpath, type=str,
                        help='Path to csv file - Default is Padchest')
    parser.add_argument('--saved_model_path', default=None, type=str,
                        help='You have to specify the path to the model to load')
    parser.add_argument('--saveplotdir', default=default_saveplotdir, type=str,
                        help='Path to the directory where plots will be saved')
    parser.add_argument('--skipfirst', default=0, type=int,
                        help='Index at which the loop will start. Allows to skip first data augmentation levels')

    args = parser.parse_args()

    pathologies = ["Atelectasis", "Consolidation", "Infiltration",
                   "Pneumothorax", "Edema", "Emphysema", "Fibrosis", "Effusion", "Pneumonia",
                   "Pleural_Thickening", "Cardiomegaly", "Nodule", "Mass", "Hernia"]

    original_paper_results = ["0.8094", "0.7901", "0.7345",
                              "0.8887", "0.8878", "0.9371", "0.8047", "0.8638", "0.7680",
                              "0.8062", "0.9248", "0.7802", "0.8676", "0.9164"]

    """
    # Local
    datadir = "/home/user1/Documents/Data/PADCHEST/sample"
    val_csvpath =  "/home/user1/Documents/Data/PADCHEST/chest_x_ray_images_labels_sample.csv"
    saved_model_path = "Models/model_178800.pth"
    saveplotdir = "/home/user1/PycharmProjects/ChestXrays/Plots/model_test"

    """
    # Server
    datadir = args.datadir
    val_csvpath = args.val_csvpath
    saved_model_path = args.saved_model_path
    saveplotdir = args.saveplotdir
    skipfirst = args.skipfirst

    if not os.path.exists(saveplotdir):
        os.makedirs(saveplotdir)

    print("Arguments :", datadir, val_csvpath, saved_model_path, saveplotdir)

    inputsize = [224, 224]  # Image Size fed to the network
    batch_size = 16
    n_batch = -1  # Number of batches used to compute the AUC, -1 for all validation set
    n_splits = 10  # Number of randomized splits to compute standard deviations
    split = ShuffleSplit(n_splits=n_splits, test_size=0.5, random_state=0)

    rot = [5, 10, 15, 25, 45, 65, 90, 180]
    translate = [(0.03, 0.03), (0.05, 0.05), (0.10, 0.10), (0.12, 0.12),
                 (0.15, 0.15), (0.25, 0.25), (0.35, 0.35), (0.50, 0.50)]
    scale = [(0.97, 1.03), (0.95, 1.05), (0.9, 1.1), (0.87, 1.13),
             (0.85, 1.15), (0.8, 1.2), (0.75, 1.2), (0.7, 1.2)]

    for data_aug in range(skipfirst, 9):

        if data_aug < 8:

            data_transforms = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomHorizontalFlip(),
                transforms.RandomAffine(rot[data_aug], translate=translate[data_aug], scale=scale[data_aug]),
                # transforms.RandomVerticalFlip(),
                transforms.Resize(inputsize),
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x.repeat(3, 1, 1))
            ])

            print("rotation", rot[data_aug], "translation", translate[data_aug], "scale", scale[data_aug])

        else:
            data_transforms = transforms.Compose([
                transforms.ToPILImage(),
                # transforms.RandomHorizontalFlip(),
                # transforms.RandomAffine(rot[data_aug], translate=translate[data_aug], scale=scale[data_aug]),
                # transforms.RandomVerticalFlip(),
                transforms.Resize(inputsize),
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x.repeat(3, 1, 1))
            ])

            print("without data augmentation")

        ################################################################################################################
        # Compute predictions
        ################################################################################################################

        # val_dataloader = MyDataLoader(datadir, val_csvpath, inputsize, batch_size=batch_size, drop_last=True,
        #                               data_transforms=data_transforms)
        val_dataloader = PadChestDataLoader(datadir, val_csvpath, inputsize, batch_size=batch_size, drop_last=True,
                                            data_transforms=data_transforms)
        # all_data = pd.read_csv(val_csvpath)

        if n_batch == -1:
            n_batch = len(val_dataloader)

        # Initialize result arrays
        all_outputs = np.zeros((n_batch * batch_size, 14))
        all_labels = np.zeros((n_batch * batch_size, 14))
        all_idx = np.zeros((n_batch * batch_size, 1))

        # Model
        if torch.cuda.is_available():
            densenet = myDenseNet().cuda()
            densenet = addDropout(densenet, p=0)
            densenet.load_state_dict(torch.load(saved_model_path))
            # If pretrained model from git repo
            # densenet = DenseNet121(14).cuda()
            # densenet.load_state_dict(load_dictionary(saved_model_path))
        else:
            densenet = myDenseNet()
            densenet = addDropout(densenet, p=0)
            densenet.load_state_dict(torch.load(saved_model_path, map_location='cpu'))
            # If pretrained model from git repo
            # densenet = DenseNet121(14)
            # densenet.load_state_dict(load_dictionary(saved_model_path, map_location='cpu'))

        cpt = 0
        densenet.eval()

        for data, label in val_dataloader:

            if torch.cuda.is_available():
                data = data.cuda()
                label = label.cuda()

            # print(np.max(data[0].detach().numpy()))

            data = data.float()
            output = densenet(data)[-1]

            if torch.cuda.is_available():
                all_labels[cpt * batch_size: (cpt + 1) * batch_size] = label.detach().cpu().numpy()
                all_outputs[cpt * batch_size: (cpt + 1) * batch_size] = output.detach().cpu().numpy()
                # all_idx[cpt * batch_size: (cpt + 1) * batch_size] = idx.detach().cpu().numpy()[:, None]
            else:
                all_labels[cpt * batch_size: (cpt + 1) * batch_size] = label.detach().numpy()
                all_outputs[cpt * batch_size: (cpt + 1) * batch_size] = output.detach().numpy()
                # all_idx[cpt * batch_size: (cpt + 1) * batch_size] = idx.detach().numpy()[:, None]

            cpt += 1
            if cpt == n_batch:
                break

        # Save predictions as a csv
        # all_names = np.array([all_data['Image Index'][idx].values for idx in all_idx])
        # csv_array = pd.DataFrame(np.concatenate((all_names, all_outputs, all_labels), axis=1))
        # column_names = ["name"] + ["prediction_"+str(i) for i in range(14)] + ["label_"+str(i) for i in range(14)]
        # csv_array.to_csv("model_predictions.csv", header=column_names, index=False)

        ################################################################################################################
        # Compute AUC
        ################################################################################################################

        print("saving !")
        np.save(os.path.join(saveplotdir, "all_labels_" + str(data_aug)), all_labels)
        np.save(os.path.join(saveplotdir, "all_outputs_" + str(data_aug)), all_outputs)

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
                plt.subplot(121)
                plt.plot(fpr, tpr)
                plt.text(0.6, 0.2, "AUC=" + str(auc)[:4])
                plt.xlabel("False Positive Rate")
                plt.ylabel("True Positive Rate")
                plt.title(pathologies[i] + " ROC")
                plt.subplot(122)
                plt.hist(all_outputs[np.where(all_labels[:, i] == 1), i][0], range=(0, 1), bins=50, label="pos")
                plt.hist(all_outputs[np.where(all_labels[:, i] == 0), i][0], range=(0, 1), bins=50, label="neg")
                plt.legend()
                plt.title(pathologies[i] + " Prediction histo")
                plt.savefig(os.path.join(saveplotdir, str(pathologies[i] + '.png')))
                plt.clf()
