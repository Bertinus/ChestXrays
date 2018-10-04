from dataset import MyDataLoader
import torch
from torch.optim import Adam
from model import myDenseNet, averageCrossEntropy


if __name__=="__main__":

    ####################################################################################################################
    # Parameters
    ####################################################################################################################

    # Dataloader
    datadir = "/home/user1/Documents/Data/ChestXray/images"
    csvpath = "/home/user1/Documents/Data/ChestXray/Data_Entry_2017.csv"
    inputsize = [224, 224]

    # Optimizer
    learning_rate = 0.001

    ####################################################################################################################
    # Training
    ####################################################################################################################

    # Dataloader
    dataloader = MyDataLoader(datadir, csvpath, inputsize)

    # Model
    densenet = myDenseNet()
    if torch.cuda.is_available():
        model = densenet().cuda()

    # Loss
    criterion = averageCrossEntropy

    # Optimizer
    optimizer = Adam(densenet.parameters(), lr=learning_rate)

    for data, label in dataloader:
        output = densenet(data)
        optimizer.zero_grad()
        loss = criterion(output, label)
        print(loss)
        loss.backward()
        optimizer.step()

