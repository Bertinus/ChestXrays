import argparse
import time
import pickle
from ALImisc import *
from ALImodel import *
from ALIloader import *
from sklearn import metrics
from sklearn import manifold

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


datadir = opt.xraydir
#ChestXray Image Dir
if os.path.exists("/data/lisa/data/ChestXray-NIHCC-2/"):
    datadir = "/data/lisa/data/ChestXray-NIHCC-2/"

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

TestDataset = LoadTrainTestSet(datadir+"ChestXray-NIHCC-2/",inputsize,rseed=13,subset="Test",N=Params["N"])
TrainDataset = LoadTrainTestSet(datadir+"ChestXray-NIHCC-2/",inputsize,rseed=13,N=len(TestDataset))
#Load MNIST
MNIST = LoadMNIST(datadir+"MNIST/",inputsize)

#Load MURA
MURA = LoadMURA(datadir+"MURA-v1.1/*/",inputsize,N=len(TestDataset),rseed=13)

#Load Pneunomia
Pneuno = LoadPneunomia(datadir+"/chest_xray/*/*",inputsize,N=len(TestDataset),rseed=13)

#Modified Chest X-ray 
Hflip,Vflip,Shuffle,Random = LoadModChest(datadir+"ChestXray-NIHCC-2/",64,rseed=13,N=len(TestDataset))

DsetName = ["ChestXray","tChestXray","MNIST","MURA","Pneuno","hFlip","vFlip","Shuffle","Random"]
Dset = [TestDataset,TrainDataset,MNIST,MURA,Pneuno,Hflip,Vflip,Shuffle,Random]

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
for cp in sorted(SavedModelsIT):
    #Load current iteration
    
    DisX,DisZ,DisXZ,GenZ,GenX,CP,DiscriminatorLoss,tAUCs = GenModel(
        inputsize,LS,cp,ExpDir,Params["name"],ColorsNumber=ColorsNumber)
    for cp in tAUCs:
        if cp in AllAUCs:continue
        for met in tAUCs[cp]:
            if cp not in AllAUCs:AllAUCs[cp] = dict()
            
            AllAUCs[cp][met] = tAUCs[cp][met]
    print("Iterations",cp,CP)
    if cp in AllAUCs:
        continue
    
    
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
            if n == "MNIST":
                ttlab = list(path.detach().numpy())
            else:
                ttlab += list(path[1])
            #Calculate Error    
            DiscSc,RL,Z,Xr,DiffX = EvalImage(GenX,GenZ,DisXZ,DisX,DisZ,rXi)
            
            #Store everything
            if torch.cuda.is_available():
                Xi = Xi.cpu()
                
            TX += list(Xi.detach().numpy())
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
    
    
    Zn = AllEvalData["tChestXray"]["Z"]
    for name in DsetName:
        tDist = []
        udist = []
        for Zi in AllEvalData[name]["Z"]:
            dist = np.sum(np.power(np.array(Zn)-Zi,2),axis=1)
            tDist.append(np.min(dist))
            udist.append(np.sum(np.power(Zi,2)))
        AllEvalData[name]["tDist"] = tDist
        AllEvalData[name]["Dist"] = udist
    
    #Print Reconstruction process
    sf = ExpDir+"/images/RecLoss/Recon_"+"%010d.png" % (cp)
    ImageReconPrint(AllEvalData,DsetName,SaveFile=sf)
    
    #Print Distribution of image
    sf = ExpDir+"/images/RecLoss/ImgDist_"+"%010d.png" % (cp)
    PrintDense(AllEvalData,DsetName,ToPrint=["ChestXray","MNIST","MURA","Pneuno"],SaveFile=sf)
    sf = ExpDir+"/images/RecLoss/SynthDist_"+"%010d.png" % (cp)
    PrintDense(AllEvalData,DsetName,ToPrint=["ChestXray","hFlip","vFlip","Shuffle","Random","tChestXray"],SaveFile=sf)
    
    
    #Print T-SNE
    df = GetTSNE(AllEvalData,ToPrint = ["ChestXray","MNIST","MURA","Pneuno","Shuffle","vFlip"])
    sf = ExpDir+"/images/RecLoss/TSNE_"+"%010d.png" % (cp)
    PrintTSNE(df,ToPrint = ["ChestXray","MNIST","MURA","Pneuno","Shuffle","vFlip"],MaxPlot=300,SaveFile=sf)
    
    #Print Sorted error
    sf = ExpDir+"/images/RecLoss/SortErr_"+"%010d.png" % (cp)
    ImageSortPrint(AllEvalData,DsetName,SaveFile=sf)
    
    #Get AUC recloss
    tAUC = GetAUC(AllEvalData,metric="RecLoss")
    if cp not in AllAUCs:
        AllAUCs[cp] = dict()
    AllAUCs[cp]["AUCrl"] = tAUC
    
    #Get AUC training dist
    tAUC = GetAUC(AllEvalData,metric="tDist")
    if cp not in AllAUCs:
        AllAUCs[cp] = dict()
    AllAUCs[cp]["AUCtDist"] = tAUC
    
    #Get AUC recloss
    tAUC = GetAUC(AllEvalData,metric="Dist")
    if cp not in AllAUCs:
        AllAUCs[cp] = dict()
    AllAUCs[cp]["AUCuDist"] = tAUC
    
    #Get AUC Discriminator
    tAUC = GetAUC(AllEvalData,metric="Dis")
    if cp not in AllAUCs:
        AllAUCs[cp] = dict()
    AllAUCs[cp]["AUCdis"] = tAUC
    
    #Get Loss for every dataset
    if "Url" not in AllAUCs[cp]:AllAUCs[cp]["Url"] = dict()
    if "Udis" not in AllAUCs[cp]:AllAUCs[cp]["Udis"] = dict()
    if "Drl" not in AllAUCs[cp]:AllAUCs[cp]["Drl"] = dict()
    if "Ddis" not in AllAUCs[cp]:AllAUCs[cp]["Ddis"] = dict()    
    for n in AllEvalData:
        AllAUCs[cp]["Url"][n] = np.mean(AllEvalData[n]["RecLoss"])
        AllAUCs[cp]["Udis"][n] = np.mean(AllEvalData[n]["Dis"])
        
        AllAUCs[cp]["Drl"][n] = np.std(AllEvalData[n]["RecLoss"])
        AllAUCs[cp]["Ddis"][n] = np.std(AllEvalData[n]["Dis"])
    
    pickle.dump(AllAUCs, open( '{0}/models/{1}_AUCs_It_{2}.pth'.format(ExpDir,Params["name"], cp), "wb" ))
    
    


        
        
