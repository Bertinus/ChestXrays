from model import *
from AliLoader import *
from ALI_Out import *
from torchvision import transforms
from torch.utils.data import DataLoader
import torchvision.datasets as dset
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import manifold
from sklearn import metrics
from scipy import stats
import argparse



parser = argparse.ArgumentParser()
#Parse command line
parser.add_argument('--epoch', type=int, default=100, help='Epoch')
parser.add_argument('--LS', type=int, default=128, help='Latent Size')
parser.add_argument('--batch-size', type=int, default=100, help='Batch Size')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--beta2', type=float, default=1e-3, help='beta2 for adam. default=0.999')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate for optimizer, default=0.00005')
parser.add_argument('--N', type=int, default=-1, help='Number of images to load (-1 for all), default=-1')
parser.add_argument('--name', type=str, default="default", help='Experiment name')
parser.add_argument('--checkpoint', type=int, default=-3, help='Checkpoint epoch to load')
parser.add_argument('--make-check',type = int, default =1,help="When to make checkpoint")
parser.add_argument('--wrkdir',type = str, default = "NA",help="Output directory of the experiment")
parser.add_argument('--eval', help="No Training",action='store_true')

opt = parser.parse_args()
print(opt)

Epoch = opt.epoch
LS = opt.LS #Latent Space Size
batch_size = opt.batch_size
ColorsNumber = 1

lr = opt.lr
b1 = opt.beta1
b2 = opt.beta2

#Directory with all model
ModelDir = "./model/"
if os.path.exists("/data/milatmp1/frappivi/ALI_model"):
    ModelDir = "/data/milatmp1/frappivi/ALI_model/"
    
if opt.wrkdir != "NA":
    if not os.path.exists(opt.wrkdir):
        os.makedirs(opt.wrkdir)
    ModelDir = opt.wrkdir
else:
    print("No --wrkdir", opt.wrkdir)
    
    
ExpDir = ModelDir+opt.name
if not os.path.exists(ExpDir):
    os.makedirs(ExpDir)
    os.makedirs(ExpDir+"/models")
    os.makedirs(ExpDir+"/images")
    
    
    
print("Wrkdir = %s" % (ExpDir))



#Encoder param
EncKernel = [5,4,4,4,4,1,1]
EncStride = [1,2,1,2,1,1,1]
EncDepth = [32,64,128,256,512,512,LS]

#Generator param
GenKernel = [4,4,4,4,5,1,1]
GenStride = [1,2,1,2,1,1,1]
GenDepth = [256,128,64,32,32,32,ColorsNumber]

#Discriminator X param
DxKernel = [5,4,4,4,4]
DxStride = [1,2,1,2,1]
DxDepth = [32,64,128,256,512]

#Discriminator Z param
DzKernel = [1,1]
DzStride = [1,1]
DzDepth = [512,512]

#Concat Discriminator param
DxzKernel = [1,1,1]
DxzStride = [1,1,1]
DxzDepth = [1024,1024,1]




#Image Dir
datadir = "./images/"
if os.path.exists("/data/lisa/data/ChestXray-NIHCC-2/images"):
    datadir = "/data/lisa/data/ChestXray-NIHCC-2/images/"



#Load data
# Transformations
inputsize = [224, 224]
inputsize = [32,32]
data_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(inputsize),
#    transforms.RandomHorizontalFlip(p=1.1),
#    transforms.RandomVerticalFlip(p=1.1),
    transforms.ToTensor(),
])

# Initialize dataloader
dataset = XrayDataset(datadir, transform=data_transforms,nrows=opt.N)

DataLen = len(dataset)
test_size = 400
train_size = DataLen-test_size

train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
ConstantImg = DataLoader(test_dataset, shuffle=False, batch_size=batch_size)
print("Dataset Len = %d" % (DataLen))

#Create Model

DisX = DiscriminatorX(KS=DxKernel,ST=DxStride,DP=DxDepth)
DisZ = DiscriminatorZ(KS=DzKernel,ST=DzStride,DP=DzDepth,LS=LS)
DisXZ = DiscriminatorXZ(KS=DxzKernel,ST=DxzStride,DP=DxzDepth)

GenZ = Encoder(KS=EncKernel,ST=EncStride,DP=EncDepth,LS=LS)
GenX = Generator(latent_size=LS,KS=GenKernel,ST=GenStride,DP=GenDepth)

#Check checkpoint to use
if opt.checkpoint == -2:
    #Find latest
    MaxCk = 0
    for fck in glob.glob('{0}/models/{1}_DisXZ_epoch_*.pth'.format(ExpDir,opt.name)):
        nck = fck.split("_")[-1].split(".")[0]
        if int(nck) > MaxCk:MaxCk = int(nck)
    opt.checkpoint = MaxCk
    print("I found this last checkpoint %d" % (opt.checkpoint))

#Check if checkpoint exist
if os.path.isfile('{0}/models/{1}_DisXZ_epoch_{2}.pth'.format(ExpDir,opt.name, opt.checkpoint)):
    print("Checkpoint %d exist, will load param and start training from there" % (opt.checkpoint))
    DisX.load_state_dict(torch.load('{0}/models/{1}_DisX_epoch_{2}.pth'.format(ExpDir,opt.name, opt.checkpoint),map_location={'cuda:0': 'cpu'}))
    DisZ.load_state_dict(torch.load('{0}/models/{1}_DisZ_epoch_{2}.pth'.format(ExpDir,opt.name, opt.checkpoint),map_location={'cuda:0': 'cpu'}))
    DisXZ.load_state_dict(torch.load('{0}/models/{1}_DisXZ_epoch_{2}.pth'.format(ExpDir,opt.name, opt.checkpoint),map_location={'cuda:0': 'cpu'}))
    
    GenZ.load_state_dict(torch.load('{0}/models/{1}_GenZ_epoch_{2}.pth'.format(ExpDir,opt.name, opt.checkpoint),map_location={'cuda:0': 'cpu'}))
    GenX.load_state_dict(torch.load('{0}/models/{1}_GenX_epoch_{2}.pth'.format(ExpDir,opt.name, opt.checkpoint),map_location={'cuda:0': 'cpu'}))
    print("Done loading")

if torch.cuda.is_available():
    print("cuda is available")
    DisX = DisX.cuda()
    DisZ = DisZ.cuda()
    DisXZ = DisXZ.cuda()

    GenZ = GenZ.cuda()
    GenX = GenX.cuda()

#Write a bunch of param about training run



#Optimiser
optimizerG = optim.Adam([{'params' : GenX.parameters()},
                         {'params' : GenZ.parameters()}], lr=lr, betas=(b1,b2))

optimizerD = optim.Adam([{'params' : DisZ.parameters()},{'params': DisX.parameters()},
                         {'params' : DisXZ.parameters()}], lr=lr, betas=(b1,b2))

DiscriminatorLoss = []

from torch.autograd import Variable

#Keep same random seed for image testing
ConstantZ = torch.randn(9,LS,1,1)
if torch.cuda.is_available():
    ConstantZ = ConstantZ.cuda()


#MNIST
MNIST_transform = transforms.Compose([transforms.Resize(inputsize),transforms.ToTensor()])
MNIST_set = dset.MNIST(root=ExpDir, train=True, transform=MNIST_transform, download=True)
MNIST_loader = torch.utils.data.DataLoader(dataset=MNIST_set,batch_size=batch_size,shuffle=False)

#Other Xray
OtherXRayDir = "/data/lisa/data/MURA-v1.1/"
OtherXRay = OtherXrayDataset("./OtherXray/", transform=data_transforms)
otherxray = DataLoader(OtherXRay, shuffle=False, batch_size=batch_size)
#Loss function
criterion = nn.BCELoss()
for epoch in range(Epoch):
    #Set to train
    GenX.train()
    GenZ.train()
    DisX.train()
    DisZ.train()
    DisXZ.train()
    
    if epoch <= opt.checkpoint:
        continue
    c = 0
    for dataiter in dataloader:
        if opt.eval == True:
            break
        #Get Data
        Xnorm = dataiter *2.0 - 1.0
        #Get Batch Size
        BS = Xnorm.shape[0]
        #Ignore weirdly small batchsize
        if BS < batch_size/2.0:
            continue
        
        FakeZ = torch.randn(BS,LS,1,1)
        #To cuda
        if torch.cuda.is_available():
            Xnorm = Xnorm.cuda()
            FakeZ = FakeZ.cuda()
        
        
        
        #Generate Fake data from random Latent
        
        FakeX = GenX(FakeZ)
        
        #Generate Latent from Real
        RealZ = GenZ(Xnorm)
        
        #Have discriminator do is thing on real and fake data
        RealCat= torch.cat((DisZ(RealZ), DisX(Xnorm)), 1)
        FakeCat= torch.cat((DisZ(FakeZ), DisX(FakeX)), 1)
        PredReal  = DisXZ(RealCat)
        PredFalse = DisXZ(FakeCat)
        
        
        #Gen fake and true label
        TrueLabel = Variable(torch.ones(BS)-0.1)
        FakeLabel = Variable(torch.zeros(BS))
        if torch.cuda.is_available():
            TrueLabel = TrueLabel.cuda()
            FakeLabel = FakeLabel.cuda()
        #Get loss for discriminator
        loss_d = criterion(PredReal.view(-1), TrueLabel) + criterion(PredFalse.view(-1), FakeLabel)

        #Get loss for generator
        loss_g = criterion(PredFalse.view(-1), TrueLabel) + criterion(PredReal.view(-1), FakeLabel)

        #Optimize Discriminator
        optimizerD.zero_grad()
        loss_d.backward(retain_graph=True)
        optimizerD.step()
    
        #Optimize Generator
        optimizerG.zero_grad()
        loss_g.backward()
        optimizerG.step()
    
        #StoreInfo .cpu().numpy()
        DiscriminatorLoss.append(loss_d.cpu().detach().numpy()+0)
        c += BS
        print("Epoch:%3d c:%6d/%6d = %6.2f Loss:%.4f" % (epoch,c,DataLen,c/float(DataLen)*100,DiscriminatorLoss[-1]))
        
    tosave = -1
    if epoch % opt.make_check == 0:
        tosave = epoch    
    # do checkpointing
    torch.save(GenX.state_dict(),
               '{0}/models/{1}_GenX_epoch_{2}.pth'.format(ExpDir,opt.name, tosave))
    torch.save(GenZ.state_dict(),
               '{0}/models/{1}_GenZ_epoch_{2}.pth'.format(ExpDir,opt.name, tosave))
    torch.save(DisX.state_dict(),
               '{0}/models/{1}_DisX_epoch_{2}.pth'.format(ExpDir,opt.name, tosave))
    torch.save(DisZ.state_dict(),
               '{0}/models/{1}_DisZ_epoch_{2}.pth'.format(ExpDir,opt.name, tosave))
    torch.save(DisXZ.state_dict(),
               '{0}/models/{1}_DisXZ_epoch_{2}.pth'.format(ExpDir,opt.name, tosave))
               
               
    #Print some image
    
    #Print generated
    
    #Set to eval
    GenX.eval()
    GenZ.eval()
    DisX.eval()
    DisZ.eval()
    DisXZ.eval()
    
    #Print Fake
    with torch.no_grad():
        FakeData = GenX(ConstantZ)
        PredFalse = DisXZ(torch.cat((DisZ(ConstantZ), DisX(FakeData)), 1))

        if torch.cuda.is_available():
            FakeData = FakeData.cpu()
            PredFalse = PredFalse.cpu()
        
        FakeData = FakeData.detach().numpy()
        PredFalse= PredFalse.detach().numpy()
    
    fig = plt.figure(figsize=(8,8))
    c = 0
    for i in range(9):
        c +=1
        #print(fd.shape)
        plt.subplot(3,3,c)
        plt.imshow(FakeData[i][0],cmap="gray")
        plt.title("Disc=%.2f" % (PredFalse[i]))
        plt.axis("off")
    fig.savefig("%s/images/%s_Gen_epoch_%d.png" % (ExpDir,opt.name,tosave))
    
    #Reconstruct real
    toprint = True
    RealDiscSc = []
    RealRecErr = []
    RealZ = []
    
    #Shuffle
    ShufDiscSc = []
    ShufRecErr = []
    ShufZ = []
    
    #Flip
    FlipDiscSc = []
    FlipRecErr = []
    FlipZ = []
    for dataiter in ConstantImg:
        ConstantX = dataiter*2.0-1.0
        if torch.cuda.is_available():
            ConstantX = ConstantX.cuda()
        DiscSc,RecErr,Z = Reconstruct(GenZ,GenX,DisX,DisZ,DisXZ,ConstantX,ExpDir,opt.name,tosave,ImageType = "Xray",Sample = 3,SaveFile=toprint)
        RealDiscSc += DiscSc
        RealRecErr += RecErr
        RealZ += Z
        
        
        #Shuffle X-ray image
        XShuffle = np.copy(ConstantX.reshape(ConstantX.shape[0],ConstantX.shape[-1]*ConstantX.shape[-1]).detach().numpy())
        np.random.shuffle(XShuffle.transpose())
        #Back to tensor
        XShuffle = torch.tensor(XShuffle)
        XShuffle = XShuffle.reshape(ConstantX.shape[0],1,ConstantX.shape[-1],ConstantX.shape[-1])
        DiscSc,RecErr,Z = Reconstruct(GenZ,GenX,DisX,DisZ,DisXZ,XShuffle,ExpDir,opt.name,tosave,ImageType = "Shuffle",Sample = 3,SaveFile=toprint)
        ShufDiscSc += DiscSc
        ShufRecErr += RecErr
        ShufZ += Z
        
        #Flip
        XFlip = np.copy(ConstantX.detach().numpy())
        XFlip = XFlip[:,:,range(ConstantX.shape[-1])[::-1],:]
        XFlip = torch.tensor(XFlip)
        DiscSc,RecErr,Z = Reconstruct(GenZ,GenX,DisX,DisZ,DisXZ,XFlip,ExpDir,opt.name,tosave,ImageType = "Flip",Sample = 3,SaveFile=toprint)
        
        FlipDiscSc += DiscSc
        FlipRecErr += RecErr
        FlipZ += Z
        
        toprint = False
        
    #Reconstruct MNIST
    toprint = True
    MNISTDiscSc = []
    MNISTRecErr = []
    MNISTZ = []
    for mnist,lab in MNIST_loader:
        Xnorm = mnist *2.0 - 1.0
        if torch.cuda.is_available():
            Xnorm = Xnorm.cuda()
        DiscSc,RecErr,Z = Reconstruct(GenZ,GenX,DisX,DisZ,DisXZ,Xnorm,ExpDir,opt.name,tosave,ImageType = "MNIST",Sample = 3,SaveFile=toprint)
        MNISTDiscSc += DiscSc
        MNISTRecErr += RecErr
        MNISTZ += Z
        if len(MNISTDiscSc) >= len(RealDiscSc):
            break
        toprint = False
    #Other Xray
    toprint = True
    OtherDiscSc = []
    OtherRecErr = []
    OtherZ = []

    for oxray in otherxray:
        Xnorm = oxray *2.0 - 1.0
        if torch.cuda.is_available():
            Xnorm = Xnorm.cuda()
        DiscSc,RecErr,Z = Reconstruct(GenZ,GenX,DisX,DisZ,DisXZ,Xnorm,ExpDir,opt.name,tosave,ImageType = "OXray",Sample = 3,SaveFile=toprint)
        OtherDiscSc += DiscSc
        OtherRecErr += RecErr
        OtherZ += Z
        if len(MNISTDiscSc) >= len(RealDiscSc):
            break
        toprint = False
    
    tsne = manifold.TSNE(n_components=2)
    Y = tsne.fit_transform(np.concatenate((RealZ,MNISTZ,ShufZ,FlipZ,OtherZ)))
    fig = plt.figure()
    minc = 0
    maxc = len(RealZ)
    plt.scatter(Y[minc:maxc,0],Y[minc:maxc,1],c="red",label="X-ray")
    minc = maxc
    maxc += len(MNISTZ)
    plt.scatter(Y[minc:maxc,0],Y[minc:maxc,1],c="blue",label="MNIST")
    minc = maxc
    maxc += len(ShufZ)
    plt.scatter(Y[minc:maxc,0],Y[minc:maxc,1],c="green",label="Shuffle")
    minc = maxc
    maxc += len(FlipZ)
    plt.scatter(Y[minc:maxc,0],Y[minc:maxc,1],c="Yellow",label="Flip")
    
    minc = maxc
    maxc += len(OtherZ)
    plt.scatter(Y[minc:maxc,0],Y[minc:maxc,1],c="Purple",label="OXray")
    
    
    plt.legend()
    fig.savefig("%s/images/%s_TSNE_epoch_%d.png" % (ExpDir,opt.name,tosave))
    
    #AUC
    AllDiscSc = [ShufDiscSc,FlipDiscSc,MNISTDiscSc,OtherDiscSc]
    AllRecErr = [ShufRecErr,FlipRecErr,MNISTRecErr,OtherRecErr]
    AllZ = [ShufZ,FlipZ,MNISTZ,OtherZ]
    Name = ["Shuffle","Flip","MNIST","OXray"]
    
    for (d,r,z,n) in zip(AllDiscSc,AllRecErr,AllZ,Name):
        yd = RealDiscSc + d
        pred = [1]*len(RealDiscSc)+[0]*len(d)
        fpr, tpr, thresholds = metrics.roc_curve(pred,yd, pos_label=1)
        auc = metrics.auc(fpr, tpr)
        print("%10s %12s %4.2f" % ("Disc",n,auc))
        
        yr = RealRecErr + r
        fpr, tpr, thresholds = metrics.roc_curve(pred,-np.array(yr), pos_label=1)
        auc = metrics.auc(fpr, tpr)
        print("%10s %12s %4.2f" % ("RecL",n,auc))
        
        yc = stats.zscore(yd) - stats.zscore(yr)
        fpr, tpr, thresholds = metrics.roc_curve(pred,yc, pos_label=1)
        auc = metrics.auc(fpr, tpr)
        print("%10s %12s %4.2f" % ("Combine",n,auc))
        
        ydis = list(np.sum(np.power(RealZ,2),axis=1)) + list(np.sum(np.power(z,2),axis=1))
        
        fpr, tpr, thresholds = metrics.roc_curve(pred,-np.array(ydis), pos_label=1)
        auc = metrics.auc(fpr, tpr)
        print("%10s %12s %4.2f" % ("Distance",n,auc))
    #Print some distribution
    AllDiscSc += [RealDiscSc]
    AllRecErr += [RealRecErr]
    AllZ += [RealZ]
    Name += ["Chest X-ray"]
    
    for (ds,tn) in zip([AllDiscSc,AllRecErr,AllZ],["Discriminator","RecLoss","Distance"]):
      fig = plt.figure()
      for (d,n) in zip(ds,Name):
          #print(np.shape(d))
          if (len(np.shape(d)) > 1):
              d = np.sum(np.power(d,2),axis=1)
          plt.hist(d,label=n, density=True,histtype="step",bins=20)
      plt.xlabel(tn)
      plt.legend()
      fig.savefig("%s/images/%s_Dist%s_epoch_%d.png" % (ExpDir,opt.name,tn,tosave))
    
    if opt.eval == True:
        break
        
                   
                   
        
       
