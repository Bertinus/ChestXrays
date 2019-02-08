import argparse
import time
import pickle
from ALImisc import *
from ALImodel import *
from ALIloader import *
from sklearn import metrics
from sklearn import manifold
from skimage.measure import compare_ssim as ssim

AllFold = glob.glob("/network/tmp1/frappivi/ALI_model/Exp_64_*")

datadir = "/media/vince/MILA/Chest_data/"
#ChestXray Image Dir
if os.path.exists("/network/data1/"):
    datadir = "/network/data1/"
inputsize = 64
N = -1
#Load train and test
print("Loading test and train")
TestDataset = LoadTrainTestSet(datadir+"ChestXray-NIHCC-2/",inputsize,rseed=13,subset="Test",N=N,verbose=1,split="old")
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




for folder in AllFold:
    if "nofinding" in folder:continue
    print(folder)
    name = folder.split("/")[-1]
    ColorsNumber = 1
    #Create all the folders to save stuff
    ExpDir,ModelDir = CreateFolder("NA",name)
    
    #Get all Modeled saved
    SavedModelsIT = []
    for SavedFiles in glob.glob('{0}/models/*_DisXZ_It_*.pth'.format(ExpDir)):
        #print(fck)
        nck = SavedFiles.split("_")[-1].split(".")[0]
        SavedModelsIT.append(int(nck))
    SavedModelsIT = sorted(SavedModelsIT)
    print("Saved Model",SavedModelsIT[-10:])
    if len(SavedModelsIT) == 0:
        continue
    
    
    
    
    
    ModelName = name
    
    

    #Print Argument
    Params = dict()
    Params["ExpDir"] = ExpDir
    Params["xraydir"] = datadir
    Params["name"] = name
    if os.path.exists(ExpDir+"/params.pk"):
        OldParams = pickle.load(open(ExpDir+"/params.pk","rb"))
        Req = ["LS","inputsize"]
        for r in Req:
            Params[r] = OldParams[r]
    else:
        continue
            
    LS = Params["LS"] #Latent Space Size
    inputsize = Params["inputsize"]

    


    #Eval every model


    AllAUCs = dict()
    for cp in sorted(SavedModelsIT)[::-1]:
        f = '{0}/{1}_AllAUC_It_{2}.pth'.format("/network/home/frappivi/chestAUCs/",name, cp)
        if os.path.exists(f):break
        #die
        #Load cu#rrent iteration
        
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
                
                
        AllAUCs = dict()
        AllAUCs["MeanL2"] = np.mean(AllEvalData["ChestXray"]["RecLoss"])
        for t in ["RecLoss","ssim","tDist","Dist","Dis"]:
            tAUC = GetAUC(AllEvalData,metric=t)
            tname = "AUC_"+t
            AllAUCs[tname] = tAUC
            print(tname)
            for k in sorted(tAUC.keys()):
                print("%10s %6.2f %6.2f" % (k,tAUC[k],1-tAUC[k])) 
        pickle.dump(AllAUCs, open( '{0}/{1}_AllAUC_It_{2}.pth'.format("/network/home/frappivi/chestAUCs/",Params["name"], cp), "wb" ))
        break
