import torchvision.models as models
from torch import nn
from torch.nn.modules.linear import Linear
import torch


def averageCrossEntropy(output, label):
    """
    :param output: Tensor of shape batchSize x Nclasses
    :param label: Tensor of shape batchSize x Nclasses
    :return: Sum of the per class entropies
    """
    loss = torch.tensor(0.)
    crossEntropy = nn.CrossEntropyLoss()
    for i in range(output.size()[1]):
        loss += crossEntropy(torch.cat((1-output[:, i:i+1], output[:, i:i+1]), 1), label[:, i])

    return loss


def myDenseNet(out_features=14):
    net = models.densenet121(pretrained=True)
    # Change last layer : fc + sigmoid
    net.classifier = nn.Sequential(Linear(in_features=1024, out_features=out_features), nn.Sigmoid())
    return net


if __name__=="__main__":

    ####################################################################################################################
    # Test
    ####################################################################################################################

    densenet = myDenseNet()

    print(type(densenet.classifier[0]), type(densenet.classifier[1]))

    for name, param in densenet.named_parameters():
        print(name, param.requires_grad)
