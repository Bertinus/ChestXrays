import argparse
import time
import pickle
from AliMisc import *
from model import *
from AliLoader import *
from ALI_Out import *
from sklearn import metrics
from sklearn import manifold

parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str, default="default", help='Experiment name')
parser.add_argument('--wrkdir',type = str, default = "NA",help="Output directory of the experiment")
parser.add_argument('--xraydir',help="Directory Chest X-Ray images",default = "./ChestXray-NIHCC-2/",type=str)
parser.add_argument('--epoch',type=int,help="Epoch to run (-2 run last,-1 run all)",default = -2)
parser.add_argument('--LS', type=int, default=128, help='Latent Size')
parser.add_argument('--inputsize',help="Size of image",default = 32,type=int)

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

#Load all data
dataloader,train_size,test_size,OtherSet,OtherName = CreateDataset(datadir,ExpDir,inputsize,-1,100,ModelDir,TestRatio=0.2,rseed=13,Testing = True)

DisX,DisZ,DisXZ,GenZ,GenX,CP,DiscriminatorLoss,AllAUCs = GenModel(opt.inputsize,LS,-2,ExpDir,opt.name,ColorsNumber=ColorsNumber)
for cp in range(CP+1):
    if (opt.epoch == -2):
        if cp != CP:
            continue
    if opt.epoch != -1:
        if (opt.epoch == -2):
            if cp != CP:
                continue
        else:
            if opt.epoch != cp:
                continue
    print("Epoch",cp)
    DisX,DisZ,DisXZ,GenZ,GenX,CP,DiscriminatorLoss,tAUCs = GenModel(opt.inputsize,LS,cp,ExpDir,opt.name,ColorsNumber=ColorsNumber)
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
        toprint = True
        print(n)
        #Store some value
        TDiscSc = []
        TRecErr = []
        TZ = []
        TX = []
        Tlab = []
        for dataiter,lab in dl:
            ConstantX = dataiter*2.0-1.0
            if torch.cuda.is_available():
                ConstantX = ConstantX.cuda()
            DiscSc,RecErr,Z = Reconstruct(GenZ,GenX,DisX,DisZ,DisXZ,ConstantX,ExpDir,opt.name,tosave,ImageType = n,Sample = 20,SaveFile=toprint)
            TDiscSc += DiscSc
            TRecErr += RecErr
            TZ += Z
            if n == "MNIST":
                if torch.cuda.is_available():
                    lab = lab.cpu()
                lab = list(lab.detach().numpy())
            else:
                lab = list(lab[1])
            #Keep image
            if torch.cuda.is_available():
                ConstantX = ConstantX.cpu()
            TX += list(ConstantX.detach().numpy())
            Tlab += list(lab)
            #print(TX)
            toprint = False
            if len(TZ) > test_size:
                TZ = TZ[:test_size]
                TX = TX[:test_size]
                TDiscSc = TDiscSc[:test_size]
                TRecErr = TRecErr[:test_size]
                break
<<<<<<< HEAD
            #break
=======
#            break
>>>>>>> 515afe1cb7452388e461d6f99171020b80eba9ea
        AllEvalData[n]["Z"] = TZ
        AllEvalData[n]["X"] = TX
        AllEvalData[n]["RecLoss"] = TRecErr
        AllEvalData[n]["Dis"] = TDiscSc
        AllEvalData[n]["Lab"] = Tlab
        print(len(Tlab))
    for n in OtherName:
        print(n)
        c = 0
        fig = plt.figure(figsize=(8,8))
        sind = np.argsort(AllEvalData[n]["RecLoss"])
        
        for ind in sind[0:50]:
            c +=1
            plt.subplot(10,10,c)
            plt.imshow(AllEvalData[n]["X"][ind][0],cmap="gray",vmin=-1,vmax=1)
            #plt.title("ReL=%.2f" % (AllEvalData[n]["RecLoss"][ind]))
            plt.axis("off")
        for ind in sind[-50:]:
            c +=1
            plt.subplot(10,10,c)
            plt.imshow(AllEvalData[n]["X"][ind][0],cmap="gray",vmin=-1,vmax=1)
            #plt.title("ReL=%.2f" % (AllEvalData[n]["RecLoss"][ind]))
            plt.axis("off")
        fig.savefig("%s/images/%s_%s_SortError_epoch_%s.png" % (ExpDir,opt.name,n,tosave))
    
    for tn in ["RecLoss","Dis"]:
      fig = plt.figure(figsize=(8,8))
      for n in OtherName:
          
          d = AllEvalData[n]["RecLoss"]
          RealDiscSc = AllEvalData["XRayT"]["RecLoss"]
          yd = RealDiscSc + d
          pred = [1]*len(RealDiscSc)+[0]*len(d)
          fpr, tpr, thresholds = metrics.roc_curve(pred,-np.array(yd), pos_label=1)
          auc = metrics.auc(fpr, tpr)
          print(tn,n,auc)
          #Calculate AUC
          plt.hist(d,label=n + " %.2f" % (auc), density=True,histtype="step",bins=20)
          plt.xlabel(tn)
      plt.legend()
      
      fig.savefig("%s/images/%s_%s_Dist%s_epoch_%s.png" % (ExpDir,tn,opt.name,tn,tosave))

    #Do T-SNE
    AllZ = []
    for n in sorted(AllEvalData.keys()):
        AllZ += AllEvalData[n]["Z"]
    Y = manifold.TSNE(n_components=2).fit_transform(AllZ)
    fig = plt.figure()
    minc = 0
    maxc = 0
    for n in sorted(AllEvalData.keys()):
        maxc += len(AllEvalData[n]["Z"])
        plt.scatter(Y[minc:maxc,0],Y[minc:maxc,1],label=n,s=0.5)
        minc = maxc
    plt.legend()
    fig.savefig("%s/images/%s_TSNE_epoch_%s.png" % (ExpDir,opt.name,tosave))
    
    #for n in ["XRayT","OXray","MNIST"]:
    for n in ["XRayT"]:
        fig = plt.figure(figsize=(14,14))
        AllZ = AllEvalData[n]["Z"]
        Y = manifold.TSNE(n_components=2).fit_transform(AllZ)
        df = pd.DataFrame([Z[0],Z[1],AllEvalData[n]["Lab"]]).transpose()
        df.columns = ["Z0","Z1","lab"]
        df[["Z0","Z1"]] = df[["Z0","Z1"]].apply(pd.to_numeric)
        #print(df["lab"].value_counts())
        #print(df.head(5))
        ind = df["lab"].value_counts().head(10).index
        for uni in ind:
            subdf = df[df["lab"] == uni]
            print(uni)
            print(subdf.head(5))
            plt.scatter(subdf["Z0"],subdf["Z1"],label=uni)
        plt.legend()
        fig.savefig("%s/images/%s_TSNE_%s_epoch_%s.png" % (ExpDir,opt.name,n,tosave))
    for n in ["XRayT"]:
<<<<<<< HEAD
        df = pd.DataFrame([AllEvalData[n]["RecLoss"],np.array(AllEvalData[n]["Lab"]) == "no_finding"]).transpose()
=======
        df = pd.DataFrame([AllEvalData[n]["RecLoss"],list(np.array(AllEvalData[n]["Lab"]) == "no_finding")]).transpose()
>>>>>>> 515afe1cb7452388e461d6f99171020b80eba9ea
        df.columns = ["RecLoss","Lab"]
        df["RecLoss"] = df["RecLoss"].apply(pd.to_numeric)
        print(df.groupby(by=["Lab"]).mean().sort_values(by="RecLoss"))
        #fpr, tpr, thresholds = metrics.roc_curve(df["Lab"],df["RecLoss"], pos_label=True)
        #auc = metrics.auc(fpr, tpr)    
        #print(auc)
        
        df = pd.DataFrame([AllEvalData[n]["RecLoss"],list(np.array(AllEvalData[n]["Lab"]) )]).transpose()
        print(df)
        
        df = df.sort_values(by="RecLoss")
        print(df)
        
        
        
