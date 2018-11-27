import argparse
import time
import pickle
from AliMisc import *
from model import *
from AliLoader import *
from ALI_Out import *
from sklearn import metrics
from sklearn import manifold

#Parse command line
parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str, default="default", help='Experiment name')
parser.add_argument('--wrkdir',type = str, default = "NA",help="Output directory of the experiment")
parser.add_argument('--xraydir',help="Directory Chest X-Ray images",default = "./ChestXray-NIHCC-2/",type=str)
parser.add_argument('--epoch',type=int,help="Epoch to run (-2 run last,-1 run all)",default = -2)
parser.add_argument('--LS', type=int, default=128, help='Latent Size')
parser.add_argument('--inputsize',help="Size of image",default = 32,type=int)

opt = parser.parse_args()

ColorsNumber = 1
#Create all the folders to save stuff (should already exist)
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

#Get all Modeled saved
SavedModelsIT = []
for SavedFiles in glob.glob('{0}/models/*_DisXZ_It_*.pth'.format(ExpDir)):
    #print(fck)
    nck = SavedFiles.split("_")[-1].split(".")[0]
    SavedModelsIT.append(int(nck))
SavedModelsIT = sorted(SavedModelsIT)
print("Saved Model",SavedModelsIT)
#Load all data
dataloader,train_size,test_size,OtherSet,OtherName = CreateDataset(datadir,ExpDir,inputsize,-1,100,ModelDir,TestRatio=0.2,rseed=13,Testing = True)
AllAUCs = dict()
for cp in sorted(SavedModelsIT)[::-1]:
    #Load current iteration
    print("Iterations",cp)
    
    #Check if exist
    
    DisX,DisZ,DisXZ,GenZ,GenX,CP,DiscriminatorLoss,tAUCs = GenModel(opt.inputsize,LS,cp,ExpDir,opt.name,ColorsNumber=ColorsNumber)
    #print(tAUCs)
    #Copy this it
    for tn in tAUCs:
        for tcp in tAUCs[tn]:
          if tn not in AllAUCs:AllAUCs[tn] = dict()
          if tcp not in AllAUCs[tn]:AllAUCs[tn][cp] = dict()
          AllAUCs[tn][tcp] = tAUCs[tn][tcp]  
    if "RecLoss" in AllAUCs:
        if cp in AllAUCs["RecLoss"]:
            continue
    #Set to eval
    GenX.eval()
    GenZ.eval()
    DisX.eval()
    DisZ.eval()
    DisXZ.eval()
    
    tosave = cp
    AllEvalData = dict()
    for (dl,n) in zip(OtherSet,OtherName):
        AllEvalData[n] = dict()
        toprint = False
        #Store some value
        TDiscSc = []
        TRecErr = []
        TZ = []
        TX = []
        for dataiter,lab in dl:
            ConstantX = dataiter*2.0-1.0
            if torch.cuda.is_available():
                dataiter = dataiter.cuda()
            DiscSc,RecErr,Z = Reconstruct(GenZ,GenX,DisX,DisZ,DisXZ,ConstantX,ExpDir,opt.name,tosave,ImageType = n,Sample = 3,SaveFile=toprint)
            TDiscSc += DiscSc
            TRecErr += RecErr
            TZ += Z
            
            #Keep image
            if torch.cuda.is_available():
                ConstantX = ConstantX.cpu()
            TX += list(ConstantX.detach().numpy())
            toprint = False
            if len(TZ) > test_size:
                TZ = TZ[:test_size]
                TX = TX[:test_size]
                TDiscSc = TDiscSc[:test_size]
                TRecErr = TRecErr[:test_size]
                break
                
        AllEvalData[n]["Z"] = TZ
        AllEvalData[n]["X"] = TX
        AllEvalData[n]["RecLoss"] = TRecErr
        AllEvalData[n]["Dis"] = TDiscSc
    for tn in ["RecLoss","Dis"]:
      for n in OtherName:
          if n == "XRayT":
              continue
          d = AllEvalData[n][tn]
          RealDiscSc = AllEvalData["XRayT"][tn]
          yd = RealDiscSc + d
          pred = [1]*len(RealDiscSc)+[0]*len(d)
          mod = 1.0
          if tn == "RecLoss":
              mod = -1.0
          fpr, tpr, thresholds = metrics.roc_curve(pred,mod*np.array(yd), pos_label=1)
          auc = metrics.auc(fpr, tpr)
          if tn not in AllAUCs:
              AllAUCs[tn] = dict()
          if cp not in AllAUCs[tn]:
              AllAUCs[tn][cp] = dict()
          AllAUCs[tn][cp][n] = auc
          print(tn,n,auc)
    pickle.dump( AllAUCs, open( '{0}/models/{1}_AUCs_It_{2}.pth'.format(ExpDir,opt.name, cp), "wb" ))
    if opt.epoch == -2:
        break
  


markers = ["",""]
col = ["red","blue","green","yellow","orange"]
linestyles = ['-', '--', '-.', ':']
for i,tn in enumerate(sorted(AllAUCs.keys())):
    subdf = pd.DataFrame(AllAUCs[tn]).transpose()
    print(subdf.max(axis=0))
    fig = plt.figure(figsize=(8,8))
    for j,n in enumerate(list(subdf.columns)):
        plt.plot(subdf.index,subdf[n],label=" ".join([tn,n]),marker=markers[i],color=col[j],linestyle=linestyles[i])
        
    plt.legend()
    fig.savefig("%s/images/AUCs_%s.png" % (ExpDir,tn))
    plt.close()         
            
        
        
        
        
    

