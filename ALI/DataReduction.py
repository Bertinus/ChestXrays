import sys
sys.path.append("../")
sys.path.append("../../")

from model import myDenseNet, addDropout, DenseNet121, load_dictionary

import argparse
import time
import pickle
from ALImisc import *
from ALImodel import *
from ALIloader import *
from skimage.measure import compare_ssim as ssim

from sklearn import metrics

def get_auc(AllEval):
    df = pd.DataFrame(AllEval).transpose().dropna()
    #print(df.columns)
    df = df[["Lab"]+sorted(df.filter(regex="_Pred").columns)+sorted(df.filter(regex="_Out").columns)]
    for pred in sorted(df.filter(regex="_Pred").columns):
        fpr, tpr, thresholds = metrics.roc_curve(df["Lab"],df[pred], pos_label=1)
        auc = metrics.auc(fpr, tpr)
        print(pred,auc)

def ScoreTensor(TensorTsc):
    ThreePeat = torch.cat((TensorTsc,TensorTsc,TensorTsc),1)
    if torch.cuda.is_available():
        ThreePeat = ThreePeat.cuda()
    Pred = densenet(ThreePeat)
    if torch.cuda.is_available():
        Pred = Pred.cpu()
    Pred = Pred.detach().numpy()
    return(Pred[:,-6])

def RunTransf(ToTest,pim,isize,GenZ,GenX,bs=20,noise=-1):
    #Tensor
    TensorTsc = torch.tensor([])
    c = 0
    trl = []
    tXr = []
    tDiffX =[]
    tAllTens = []
    tXi = ToTest
    tSSIM = []
    for i in range(len(ToTest)):

        FullImg,TensorImg = TransformPImg(pim,isize,ToTest[i])
        TensorTsc = torch.cat((TensorTsc,TensorImg*2.0-1.0),0)
        c += 1
        if c >= bs:
            #Get Score
            if torch.cuda.is_available():
                TensorTsc = TensorTsc.cuda()
            ttrl,ttXr,ttDiffX,ttssim = OutScoreRec(GenZ,GenX,TensorTsc,noise=noise)
            
            #Store everyghin
            trl += ttrl
            #tXr += list(ttXr)
            #tDiffX += list(ttDiffX)
            #tAllTens += list(TensorTsc.detach().numpy())
            tSSIM += ttssim
            #tXi += ToTest
            
            
            TensorTsc = torch.tensor([])
    if len(TensorTsc) > 0:
        ttrl,ttXr,ttDiffX,ttssim = OutScoreRec(GenZ,GenX,TensorTsc,noise=noise)

        #Store everyghin
        trl += ttrl
        #tXr += list(ttXr)
        #tDiffX += list(ttDiffX)
        #tAllTens += list(TensorTsc.detach().numpy())
        tSSIM += ttssim
            
    return(trl,tXi,tSSIM)

def GridSearch(pim,Trans=0.2,Scale=0.8,Degree=15,grid=11,nopadding= 0,noise=-1,it=1):
    ImageSize = np.max(np.shape(pim))

    AllTest = []
    TransTrans = int(ImageSize*Trans)
    
    Bounds = []
    Bounds.append([-TransTrans,TransTrans])
    Bounds.append([-TransTrans,TransTrans])
    Bounds.append([-Degree,Degree])
    Bounds.append([Scale*ImageSize,ImageSize])
    
    tXi = []
    tSSIM = []
    trl = []
    for i in range(it):
        TransfTest = []
        TransfTest.append([0,0,0,ImageSize])
        LinSpace = []
        for b in Bounds:
            tspace = list(np.linspace(b[0],b[1],grid))+[0]
            
            if b[0] == b[1]:
                tspace = [b[0]]
            LinSpace.append(sorted(list(np.unique(np.rint(tspace)))))
        for xt in LinSpace[0]:
            for yt in LinSpace[1]:
                for rot in LinSpace[2]:
                    for sc in (LinSpace[3]):
                        if sc == 0:continue
                        TransfTest.append([int(xt),int(yt),int(rot),int(sc)])





        #Eliminate redun                
        ToTest = []
        for r in TransfTest:
            #if r[3]
            if r in ToTest:continue 
            if r in AllTest:continue
            ToTest.append(r)
            AllTest.append(r)
        print(i,len(ToTest),len(TransfTest))
        if len(ToTest) == 0:
            continue
        ttrl,ttXi,ttSSIM = RunTransf(ToTest,pim,isize,GenZ,GenX,noise=noise)
        #PrintImageBound(pim,ttXi,ttrl,ttSSIM,title=i)
        
        trl += ttrl
        tXi += ttXi
        tSSIM += ttSSIM
        
        bx = tXi[np.argsort(trl)[0]]
        if i % 2 == 0:
            bx = tXi[np.argsort(tSSIM)[0]]
        nBound = []
        for j in range(4):
            if len(LinSpace[j]) == 1:
                nBound.append(Bounds[j])
                continue
            indice = np.argsort(np.abs(np.array(LinSpace[j])-bx[j]))[0]
            if indice == 0:
                indice = 1
            if indice == len(LinSpace[j])-1:
                indice = len(LinSpace[j])-2
            #print(bx[j],LinSpace[j],indice)
            nBound.append([LinSpace[j][indice-1],LinSpace[j][indice+1]])
            #print(nBound[-1])
            #nBound.append(nb)
        #die
        Bounds = list(nBound)
        
       
    return(tXi,-np.array(tSSIM),trl)

#Apply Transformation to PIM image
def TransformPImg(pim,inputsize,ar):
    tx = ar[0]
    ty = ar[1]
    rot = ar[2]
    sc = ar[3]
    brightness_factor =1.0
    contrast_factor = 1.0
    tim = transforms.functional.affine(pim,angle=rot, translate=[tx,ty], 
                                                   scale=1, shear=0, resample=0, fillcolor=0)
    
    #tim = transforms.functional.adjust_brightness(tim, brightness_factor)
    #tim = transforms.functional.adjust_contrast(tim, contrast_factor)
    data_transforms = transforms.Compose([transforms.CenterCrop(sc)])
    ftim = data_transforms(tim)
    reim = transforms.functional.resize(ftim,inputsize)
    tim = transforms.functional.to_tensor(reim)
    
    
    
    tim = tim.reshape(1,1,inputsize,inputsize)
    return(ftim,tim)
    
#Outlier score  
def OutScoreRec(GenZ,GenX,X,noise=-1):
    Noise = torch.randn(X.shape)*noise
    if torch.cuda.is_available():
        X = X.cuda()
        Noise = Noise.cuda()
    #Generate Latent from Real
    if noise > 0:
        z = GenZ(X+Noise)
    else:
        z = GenZ(X)
    #z = GenZ(X)
    Xr = GenX(z)
    
    
    DiffX = Xr - X

    
    if torch.cuda.is_available():
        DiffX = DiffX.cpu()
        X = X.cpu()
        Xr = Xr.cpu()
    DiffX = DiffX.detach().numpy()
    DiffX = np.power(DiffX,2)
    RecLoss = [np.sqrt(np.mean(x)) for x in DiffX]
    X = X.detach().numpy()
    Xr = Xr.detach().numpy()
    SSIM = []
    for i in range(len(X)):
        sd = ssim(X[i][0],Xr[i][0])
        SSIM.append(sd)
    return(RecLoss,Xr,DiffX,SSIM)    
    
def LoadPIM(ptf):
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
        transforms.Pad(padding,fill=0),
    ])
    pim = data_transforms(im)
    return(pim)    
    
    
    
    
    
    
    
densenet = DenseNet121(14)
# densenet = addDropout(densenet, p=0)
saved_model_path = "../Models/model.pth.tar"
densenet.load_state_dict(load_dictionary(saved_model_path, map_location='cpu'))    
if torch.cuda.is_available():
    densenet = densenet.cuda()    
    
name = "Exp_64_512_0.00001_RandomLabel_4.0"
#Path to the experiment (it would be the github)
ExpDir = "/media/vince/MILA/ChestXrays/ALI/model/"+name
ExpDir = "/network/home/frappivi/ChestXrays/ALI/model/"+name

isize = 64
LS = 512 #Latent Space Size
ColorsNumber = 1 #Number of color (always 1 for x-ray)
isize = 64
#Load model
CP = -2 #Checkpoint to load (-2 for latest one, -1 for last epoch)
DisX,DisZ,DisXZ,GenZ,GenX,CP,DiscriminatorLoss,AllAUCs = GenModel(isize,LS,-2,ExpDir,name,ColorsNumber=ColorsNumber)

DisX = DisX.eval()
DisZ = DisZ.eval()
DisXZ = DisXZ.eval()
GenZ = GenZ.eval()
GenX = GenX.eval()

#Get file (for the example)
datadir = "/network/data1/"
Paths = glob.glob(datadir+"ChestXray-NIHCC-2/other/chest_xray/*/*/*")


AllEval = pickle.load(open( '{0}/{1}_AllEval.pth'.format("/network/home/frappivi/ChestXrays/ALI/results/","Penumo"), "rb" ))
for i in range(len(Paths)):
    ptf = Paths[i]
    k = "/".join(ptf.split("/")[-3:]).split(".")[0]
    if k in AllEval:continue
    AllEval[k] = dict()
    print(i,len(Paths),k)
    if "PNEUMONIA" in ptf:
        AllEval[k]["Lab"] = 1
    else:
        AllEval[k]["Lab"] = 0
    pim = LoadPIM(ptf)
    
    
    
    TensorTsc = torch.tensor([])
    
    Test = [0, 0, 0, np.max(np.shape(pim))*1.0, 1.0, 1.0]
    FullImg,TensorImg = TransformPImg(pim,224,Test)
    TensorTsc = torch.cat((TensorTsc,TensorImg),0)
    
    #Find Best
    Xi,SSIM,L2 = GridSearch(pim,grid=5,Scale=0.1,Degree=15,Trans=0.2,noise=0.15,it=4)
    for nopt,name in zip([L2,SSIM],["L2","SSIM"]):
        
        ind = np.argsort(nopt)[0]
        xb = Xi[ind]
        yb = nopt[ind]
        FullImg,TensorImg = TransformPImg(pim,224,xb)
        TensorTsc = torch.cat((TensorTsc,TensorImg),0)
        
        #Store stuff
        AllEval[k][name+"_Out"] = yb
        AllEval[k][name+"_Init_Out"] = nopt[0]
        
    tPred = ScoreTensor(TensorTsc)
    AllEval[k]["Init_Pred"] = tPred[0]
    AllEval[k]["L2_Pred"] = tPred[1]
    AllEval[k]["SSIM_Pred"] = tPred[2]
    pickle.dump(AllEval, open( '{0}/{1}_AllEval.pth'.format("/network/home/frappivi/ChestXrays/ALI/results/","Penumo"), "wb" ))
    if i % 10 == 0:
        get_auc(AllEval)




