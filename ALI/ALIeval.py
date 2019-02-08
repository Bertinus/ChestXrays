import argparse
import time
import pickle
from ALImisc import *
from ALImodel import *
from ALIloader import *
from sklearn import metrics
from sklearn import manifold
from skimage.measure import compare_ssim as ssim

parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str, default="default", help='Experiment name')
parser.add_argument('--wrkdir',type = str, default = "NA",help="Output directory of the experiment")
parser.add_argument('--xraydir',help="Directory Chest X-Ray images",default = "/media/vince/MILA/Chest_data/",type=str)
parser.add_argument('--epoch',type=int,help="Epoch to run (-2 run last,-1 run all)",default = -2)
parser.add_argument('--LS', type=int, default=128, help='Latent Size')
parser.add_argument('--inputsize',help="Size of image",default = 32,type=int)
parser.add_argument('--N', type=int, default=-1, help='Number of images to load (-1 for all), default=-1')

opt = parser.parse_args()

ColorsNumber = 1
#Create all the folders to save stuff
ExpDir,ModelDir = CreateFolder(opt.wrkdir,opt.name)
ModelName = opt.name

datadir = opt.xraydir
#ChestXray Image Dir
if os.path.exists("/network/data1/"):
    datadir = "/network/data1/"

#Print Argument
Params = vars(opt)
Params["ExpDir"] = ExpDir
Params["xraydir"] = datadir
if os.path.exists(ExpDir+"/params.pk"):
    OldParams = pickle.load(open(ExpDir+"/params.pk","rb"))
    Req = ["LS","inputsize"]
    for r in Req:
        Params[r] = OldParams[r]
        
LS = Params["LS"] #Latent Space Size
inputsize = Params["inputsize"]

#Load train and test
print("Loading test and train")
TestDataset = LoadTrainTestSet(datadir+"ChestXray-NIHCC-2/",inputsize,rseed=13,subset="Test",N=Params["N"],verbose=1,split="old")
#print("TestDataset",len(TestDataset))
TrainDataset = LoadTrainTestSet(datadir+"ChestXray-NIHCC-2/",inputsize,rseed=13,N=len(TestDataset),verbose=1,split="old")
#Load MNIST
print("Loading MNIST")
MNIST = LoadMNIST(datadir+"ChestXray-NIHCC-2/other/MNIST/",inputsize)

#Load MURA
print("Loading MURA")
MURA = LoadMURA(datadir+"MURA-v1.1/MURA-v1.1/",inputsize,N=len(TestDataset),rseed=13,verbose=1)

print("Loading CIFAR")
CIFAR = LoadCIFAR(datadir+"ChestXray-NIHCC-2/other/MNIST/",inputsize)

#Load Pneunomia
print("Loading Pneumo")
Pneuno = LoadPneunomia(datadir+"ChestXray-NIHCC-2/other/chest_xray/",inputsize,N=len(TestDataset),rseed=13)

#Modified Chest X-ray 
print("Loading Synth")
Hflip,Vflip,Shuffle,Random,Mean = LoadModChest(datadir+"ChestXray-NIHCC-2/",64,rseed=13,N=len(TestDataset))

DsetName = ["ChestXray","tChestXray","CIFAR","MNIST","MURA","Pneuno","hFlip","vFlip","Shuffle","Random","Mean"]
Dset = [TestDataset,TrainDataset,CIFAR,MNIST,MURA,Pneuno,Hflip,Vflip,Shuffle,Random,Mean]

#DsetName = ["ChestXray","MNIST","MURA","Pneuno","hFlip","vFlip","Shuffle","Random","Mean"]
#Dset = [TestDataset,MNIST,MURA,Pneuno,Hflip,Vflip,Shuffle,Random,Mean]
for (d,n) in zip(Dset,DsetName):
    print(n,len(d))


#Get all Modeled saved
SavedModelsIT = []
for SavedFiles in glob.glob('{0}/models/*_DisXZ_It_*.pth'.format(ExpDir)):
    #print(fck)
    nck = SavedFiles.split("_")[-1].split(".")[0]
    SavedModelsIT.append(int(nck))
SavedModelsIT = sorted(SavedModelsIT)
print("Saved Model",SavedModelsIT)


#Eval every model


AllAUCs = dict()
for cp in sorted(SavedModelsIT)[::-1]:
    #Load current iteration
    
    DisX,DisZ,DisXZ,GenZ,GenX,CP,DiscriminatorLoss,tAUCs = GenModel(
        inputsize,LS,cp,ExpDir,Params["name"],ColorsNumber=ColorsNumber)
    cp = CP
    
    print("Iterations",cp,CP)
    #if cp in AllAUCs:
    #    continue
    
    
    #Set to eval
    GenX.eval()
    GenZ.eval()
    DisX.eval()
    DisZ.eval()
    DisXZ.eval()
    
    #Where to Store data
    AllEvalData = dict()
    for (d,n) in zip(Dset,DsetName):
        print(n)
        AllEvalData[n] = dict()
        #Array to store data
        TDiscSc = []
        TRecErr = []
        TZ = []
        TX = []
        TXr = []
        TDiff = []
        tlab = []
        for Xi,path in DataLoader(d,shuffle=False, batch_size=50):
            rXi = Xi*2.0-1.0
            if torch.cuda.is_available():
                rXi = rXi.cuda()
            ttlab = []
            if (n == "MNIST") or (n=="CIFAR"):
                ttlab = list(path.detach().numpy())
            else:
                ttlab += list(path[1])
            #Calculate Error    
            DiscSc,RL,Z,Xr,DiffX = EvalImage(GenX,GenZ,DisXZ,DisX,DisZ,rXi)
            
            #Store everything
            if torch.cuda.is_available():
                rXi = rXi.cpu()
                
            TX += list(rXi.detach().numpy())
            TDiscSc += DiscSc
            TRecErr += RL
            TZ += Z
            TXr += list(Xr)
            TDiff += list(DiffX)
            tlab += ttlab
            test_size = len(TestDataset)
            if len(TZ) >= test_size:
                TZ = TZ[:test_size]
                TX = TX[:test_size]
                TDiscSc = TDiscSc[:test_size]
                TRecErr = TRecErr[:test_size]
                TXr  = TXr[:test_size]
                TDiff  = TDiff[:test_size]
                tlab = tlab[:test_size]
                break
    
        AllEvalData[n]["Z"] = TZ
        AllEvalData[n]["X"] = TX
        AllEvalData[n]["RecLoss"] = TRecErr
        AllEvalData[n]["Dis"] = TDiscSc
        AllEvalData[n]["Xr"] = TXr
        AllEvalData[n]["DiffX"] = TDiff
        AllEvalData[n]["lab"] = tlab
    
    #Distance in latent sapce
    Zn = AllEvalData["tChestXray"]["Z"]
    if len(Zn) > 1000:
        Zn = Zn[:1000]
    for name in DsetName:
        tDist = []
        udist = []
        for Zi in AllEvalData[name]["Z"]:
            dist = np.sum(np.power(np.array(Zn)-Zi,2),axis=1)
            tDist.append(np.min(dist))
            udist.append(np.sum(np.power(Zi,2)))
        AllEvalData[name]["tDist"] = tDist
        AllEvalData[name]["Dist"] = udist
    #Get AUCs
    cp = int(CP)
    if cp not in AllAUCs:
        AllAUCs[cp] = dict()
    #Get SSIM distance
    for n in DsetName:
        ssimd = []
        x = AllEvalData[n]["X"]
        xr = AllEvalData[n]["Xr"]
        for i in range(len(x)):
            sd = ssim(x[i][0],xr[i][0])
            ssimd.append(sd)
            AllEvalData[n]["ssim"] = ssimd
            
        #Get average value
        if "u_RecLoss" not in AllAUCs[cp]:AllAUCs[cp]["u_RecLoss"] = dict()
        if "u_ssim" not in AllAUCs[cp]:AllAUCs[cp]["u_ssim"] = dict()    
        AllAUCs[cp]["u_RecLoss"][n] = np.mean(AllEvalData[n]["RecLoss"])
        AllAUCs[cp]["u_ssim"][n] = np.mean(AllEvalData[n]["ssim"])
            
            
    pickle.dump(AllEvalData, open( '{0}/{1}_AllEval_It_{2}.pth'.format(ExpDir,Params["name"], cp), "wb" ))
    for t in ["RecLoss","ssim","tDist","Dist","Dis"]:
        tAUC = GetAUC(AllEvalData,metric=t)
        tname = "AUC_"+t
        AllAUCs[cp][tname] = tAUC
        print(tname)
        for k in sorted(tAUC.keys()):
            print("%10s %6.2f %6.2f" % (k,tAUC[k],1-tAUC[k])) 
    
    #Print Reconstruction process
    sf = ExpDir+"/images/RecLoss/Recon_"+"%010d.png" % (cp)
    print(sf)
    ImageReconPrint(AllEvalData,DsetName,ToPrint=["ChestXray","MNIST","MURA","Pneuno","CIFAR"],SaveFile=sf)
    
    #Print Distribution of image
    for t in ["RecLoss","ssim","Dist"]:
        print(name,t,int(cp))
        sf = ExpDir+"/images/RecLoss/%s_Dense_%s_%010d.png" % (ModelName,t,int(cp))
        PrintDense(AllEvalData,DsetName,ToPrint=["ChestXray","MNIST","MURA","Pneuno","CIFAR"],SaveFile=sf,metric=t)
        
        
        sf = ExpDir+"/images/RecLoss/%s_Sort_%s_%010d.png" % (ModelName,t,int(cp))
        ImageSortPrint(AllEvalData,DsetName,ToPrint=["ChestXray","MNIST","MURA","Pneuno","CIFAR"],SaveFile=sf,metric=t)
    
    
    #Print T-SNE
    #df = GetTSNE(AllEvalData,ToPrint = ["ChestXray","MNIST","MURA","Pneuno","Shuffle","vFlip"])
    #sf = ExpDir+"/images/RecLoss/TSNE_"+"%010d.png" % (cp)
    #PrintTSNE(df,ToPrint = ["ChestXray","MNIST","MURA","Pneuno","Shuffle","vFlip"],MaxPlot=300,SaveFile=sf)
    
    #Print Sorted error
    
    
    
    pickle.dump(AllAUCs, open( '{0}/models/{1}_AUCs_It_{2}.pth'.format(ExpDir,Params["name"], cp), "wb" ))
    if opt.epoch == -1:
        break
