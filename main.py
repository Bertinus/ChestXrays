from dataset import MyDataLoader, Iterator
import torch
import os
from torch.optim import Adam
from model import myDenseNet, averageCrossEntropy, addDropout
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau


def initWriter(savemodeldir, logdir):
    """
    Initialize tensorboard logs
    :param savemodeldir: directory wher model weights will be saved
    :param logdir: directory where training logs will be saved
    :return: tensorboard writer
    """
    if not os.path.exists(savemodeldir):
        os.makedirs(savemodeldir)

    if not os.path.exists(logdir):
        os.makedirs(logdir)
    else:
        if os.listdir(logdir):
            print("You have to empty " + logdir)

    writer = SummaryWriter(logdir)

    return writer


def writeImages(writer, activations):
    """
    Write activations as images in tensorboard
    """

    writer.add_image('Activations/Activation_' + str(9), activations[9][0, 3:6, :, :], num_iteration)
    writer.add_image('Activations/Activation_' + str(11), activations[11][0, 3:6, :, :], num_iteration)
    # writer.add_image('Activations/Final', activations[12][0, :].view(32, 32), num_iteration)
    writer.add_image('Activations/BatchOutput', activations[13][None, :, :], num_iteration)
    writer.add_image('Activations/BatchLabels', label, num_iteration)

    writer.add_image('Weights/denseblock4.denselayer16.conv2',
                     densenet.features.denseblock4.denselayer16.conv2[0].weight[:16, :16, :, 0].transpose(0, 2),
                     num_iteration)
    writer.add_image('Weights/denseblock3.denselayer24.conv2',
                     densenet.features.denseblock3.denselayer24.conv2[0].weight[:16, :16, :, 0].transpose(0, 2),
                     num_iteration)
    writer.add_image('Weights/denseblock2.denselayer12.conv2',
                     densenet.features.denseblock2.denselayer12.conv2[0].weight[:16, :16, :, 0].transpose(0, 2),
                     num_iteration)
    writer.add_image('Weights/denseblock1.denselayer6.conv2',
                     densenet.features.denseblock1.denselayer6.conv2[0].weight[:16, :16, :, 0].transpose(0, 2),
                     num_iteration)

    # writer.add_embedding(activations[12], global_step=num_iteration)


if __name__ == "__main__":

    ####################################################################################################################
    # Parameters
    ####################################################################################################################
    """
    # Local Dataloader
    datadir = "/home/user1/Documents/Data/ChestXray/images"
    train_csvpath = "/home/user1/Documents/Data/ChestXray/DataTrain.csv"
    val_csvpath = "/home/user1/Documents/Data/ChestXray/DataVal.csv"

    # Local Writer
    savemodeldir = "/home/user1/PycharmProjects/ChestXrays/Logs/model_1"
    logdir = "/home/user1/PycharmProjects/ChestXrays/Logs/training_1"
    print("\ntensorboard --logdir=" + logdir + " --port=11995\n")
    """

    # Server Dataloader
    datadir = "/network/data1/ChestXray-NIHCC-2/images"
    train_csvpath = "/network/home/bertinpa/Documents/ChestXrays/Data/DataTrain.csv"
    val_csvpath = "/network/home/bertinpa/Documents/ChestXrays/Data/DataVal.csv"

    # Server Writer
    savemodeldir = "/network/tmp1/bertinpa/Logs/model_3"
    logdir = "/network/tmp1/bertinpa/Logs/training_3"


    # Network
    inputsize = [224, 224]
    dropout = True
    P_drop = 0.  # Original paper : 0.2

    # Number of images in the train dataset
    nrows = None  # None for the whole dataset

    # Optimizer
    learning_rate = 0.0001

    # scheduler
    sched_step_size = 10
    sched_gamma = 0.1

    # Training
    batch_size = 16
    num_epochs = 100
    val_every_n_iter = 200
    batch_per_val_session = 10
    add_graph = 1

    ####################################################################################################################
    # Initialization
    ####################################################################################################################

    print("Initializing...")

    # Dataloaders
    train_dataloader = MyDataLoader(datadir, train_csvpath, inputsize, batch_size=batch_size, nrows=nrows, flip=False)
    val_dataloader = MyDataLoader(datadir, val_csvpath, inputsize, batch_size=batch_size, flip=False)

    # Model
    if torch.cuda.is_available():
        densenet = myDenseNet().cuda()
    else:
        densenet = myDenseNet()

    # Add dropout
    if dropout:
        densenet = addDropout(densenet, p=P_drop)

    # Writer
    writer = initWriter(savemodeldir, logdir)

    # Loss
    criterion = torch.nn.BCELoss(size_average=True)  # averageCrossEntropy

    # Optimizer
    optimizer = Adam(densenet.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-5)
    # scheduler = ReduceLROnPlateau(optimizer, factor=0.1, patience=5, mode='min')
    scheduler = StepLR(optimizer, step_size=sched_step_size, gamma=sched_gamma)  # Used to decay learning rate

    ####################################################################################################################
    # Training
    ####################################################################################################################

    print("Training...")

    num_iteration = 0  # Number of iterations
    val_iterator = Iterator(val_dataloader)  # Iterator for validation samples

    for epoch in range(num_epochs):

        scheduler.step()

        # Training
        for data, label, idx in train_dataloader:

            if torch.cuda.is_available():
                data = data.cuda()
                label = label.cuda()

            # Add graph to tensorboard
            if add_graph == 1:
                add_graph = 0
                writer.add_graph(densenet, data)

            # Forward
            output = densenet(data)[-1]
            optimizer.zero_grad()
            loss = criterion(output, label)

            # Save loss
            writer.add_scalar('Train_Loss', loss, num_iteration)

            # Backward
            loss.backward()
            optimizer.step()
            num_iteration += 1

            # Validation
            if num_iteration % val_every_n_iter == 0:

                densenet.eval()
                # writeImages(writer, activations=densenet(data))
                test_loss = torch.zeros(1, requires_grad=False)

                if torch.cuda.is_available():
                    test_loss = test_loss.cuda()

                for _ in range(batch_per_val_session):

                    data, label, idx = val_iterator.next()

                    if torch.cuda.is_available():
                        data = data.cuda()
                        label = label.cuda()

                    output = densenet(data)[-1]
                    test_loss += criterion(output, label).data

                test_loss /= batch_per_val_session
                print("test", num_iteration, ":", test_loss)

                writer.add_scalar('Test_Loss', test_loss, num_iteration)

                # Save model
                torch.save(densenet.state_dict(),
                           os.path.join(savemodeldir, 'model_' + str(num_iteration) + '.pth'))

                densenet.train()
