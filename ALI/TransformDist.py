from model import *
from AliLoader import *
from ALI_Out import *

from torch.utils.data import DataLoader
import torchvision.datasets as dset
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import manifold
from sklearn import metrics
from scipy import stats
from AliMisc import *
from PIL import Image
from skopt.space import Real, Integer
from skopt import Optimizer
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--seed',help="Random Seed",default = 13,type=int)
opt = parser.parse_args()
newseed = opt.seed+np.random.randint(1,1000)
print(opt.seed,newseed)
np.random.seed(newseed)
LS = 512 #Latent Space Size
ColorsNumber = 1 #Number of color (always 1 for x-ray)
isize = 64

name = "Exp_64_512_0.00001_RandomLabel_4.0"

datadir = "./ChestXray-NIHCC-2/"

ExpDir = "./model/"+name


batch_size = 10


def GetAlphaRedMap(cmap):
    #Print Rebuild
    # Get the colormap colors
    AlphaRed = cmap(np.arange(cmap.N))
    # Set alpha
    AlphaRed[:,-1] = np.linspace(0, 1, cmap.N)
    # Create new colormap
    AlphaRed = ListedColormap(AlphaRed)
    return(AlphaRed)
AlphaRed = GetAlphaRedMap(plt.cm.Reds)

CP = -2 #Checkpoint to load (-2 for latest one, -1 for last epoch)
DisX,DisZ,DisXZ,GenZ,GenX,CP,DiscriminatorLoss,AllAUCs = GenModel(isize,LS,-2,ExpDir,name,ColorsNumber=ColorsNumber)

DisX = DisX.eval()
DisZ = DisZ.eval()
DisXZ = DisXZ.eval()
GenZ = GenZ.eval()
GenX = GenX.eval()


def TransformPImg(pim,inputsize,ar):
    tx = ar[0]
    ty = ar[1]
    rot = ar[2]
    sc = ar[3]
    brightness_factor = ar[4]
    contrast_factor = ar[5]
    tim = transforms.functional.affine(pim,angle=rot, translate=[tx,ty], 
                                                   scale=1, shear=0, resample=0, fillcolor=0)
    
    tim = transforms.functional.adjust_brightness(tim, brightness_factor)
    tim = transforms.functional.adjust_contrast(tim, contrast_factor)
    data_transforms = transforms.Compose([transforms.CenterCrop(sc)])
    ftim = data_transforms(tim)
    reim = transforms.functional.resize(ftim,inputsize)
    tim = transforms.functional.to_tensor(reim)
    tim = tim.reshape(1,1,inputsize,inputsize)
    return(ftim,tim)
    




def OutScore(DisX,DisZ,DisXZ,GenZ,GenX,X):
    z = GenZ(X)
    Xr = GenX(z)
    
    
    DiffX = Xr - X
    
    if torch.cuda.is_available():
        DiffX = DiffX.cpu()
    
    DiffX = DiffX.detach().numpy()
    DiffX = np.power(DiffX,2)
    RecLoss = [np.sqrt(np.mean(x)) for x in DiffX]

    return(RecLoss,Xr.detach().numpy(),DiffX)



Paths = glob.glob("chest_xray/*/*/*.jpeg")

RandInt = 10
Explore = 10
Paths = np.random.permutation(Paths)
for p in Paths:
    ptf = p
    print(ptf)
    
    name = "Align_"+"_".join(ptf.split("/")[-2:])
    print(name)
    if  os.path.isfile("./test/"+name):
        continue
    im = misc.imread(ptf)
    if len(im.shape) > 2:
        im = im[:, :, 0]
    #Add color chanel
    im = im[:,:,None]

    padding = 0
    if im.shape[0] > im.shape[1]:
        padding = (int((im.shape[0]-im.shape[1])/2),0)
    else:
        padding = (0,int((im.shape[1]-im.shape[0])/2))

    data_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Pad(padding,fill=0)
    ])
    pim = data_transforms(im)
    
    #pim = transforms.functional.to_pil_image(im)
    
    space  = [Integer(-800, 800, name='tx'),
          Integer(-800, 800, name='ty'),
          Real(-180,180, name='rot'),
          Integer(int(np.max(im.shape)/4), np.max(im.shape), name='sc'),
          Real(0.00001,11.4, name='Bright'),
          Real(0.00001,11.4, name='Contrast')
          ]
   
    
    
    opt = Optimizer(space)

    #Get random value
    
    
    Fimg = []
    Errs = []
    fig = plt.figure(figsize=(20,20))
    c = 1
    for it in range(Explore):
        RandTest = []
        if it == 0:
            RandTest.append([0,0,0,np.max(im.shape),1,1])
            RandTest.append([0,0,0,np.max(im.shape)/2.0,1,10])
            RandTest.append([0,0,0,np.max(im.shape)/2.0,1,0.001])
            RandTest.append([0,0,0,np.max(im.shape)/2.0,10,1])
            RandTest.append([0,0,0,np.max(im.shape)/2.0,0.001,1])
        
        totest = opt.ask(n_points=RandInt)
        RandTest += totest
        TensorTsc = torch.tensor([])
        for i in range(RandInt):
            FullImg,TensorImg = TransformPImg(pim,isize,RandTest[i])
            Fimg.append(FullImg)
            TensorTsc = torch.cat((TensorTsc,TensorImg),0)
        if torch.cuda.is_available():
            TensorTsc = TensorTsc.cuda()
        rl,Xr,DiffX = OutScore(DisX,DisZ,DisXZ,GenZ,GenX,TensorTsc*2.0-1.0)
        for i in np.argsort(rl):
            opt.tell(RandTest[i], rl[i])
            print(rl[i],RandTest[i])
            
            plt.subplot(Explore,RandInt*3,c)
            plt.imshow(TensorTsc[i][0],cmap="gray")
            #plt.imshow(DiffX[i][0],cmap=AlphaRed,vmax=4)
            
            
            #plt.title("%.2f" % (rl[i]))
            plt.axis("off")
            c += 1
            
            plt.subplot(Explore,RandInt*3,c)
            plt.imshow(Xr[i][0],cmap="gray")
            plt.title("%.2f" % (rl[i]))
            plt.axis("off")
            c += 1
            plt.subplot(Explore,RandInt*3,c)
            plt.imshow(DiffX[i][0],cmap="Reds",vmax=4)
            #plt.title("Error:%.2f" % (rl[i]))
            plt.axis("off")
            c += 1
        Errs += rl
        tsort = np.argsort(Errs)
        Errs = [Errs[tsort[0]]]
        Fimg = [Fimg[tsort[0]]]
        BestXi = opt.Xi[np.argsort(opt.yi)[0]]
        print(it,BestXi,np.sort(opt.yi)[0])
    fig.savefig("./test/test.png")
    plt.close() 
    die
    Fimg[0].save("./test/"+name, "JPEG")    
    
    
    
