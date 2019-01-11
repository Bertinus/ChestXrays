import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from matplotlib.colors import ListedColormap
from sklearn import metrics
from sklearn import manifold


#Generate Alpha Red Map (so transparent) for error
def GetAlphaRedMap():
    #Print Rebuild
    # Get the colormap colors
    cmap = plt.cm.Reds
    AlphaRed = cmap(np.arange(cmap.N))
    # Set alpha
    AlphaRed[:,-1] = np.linspace(0, 1, cmap.N)
    # Create new colormap
    AlphaRed = ListedColormap(AlphaRed)
    return(AlphaRed)
    
def GenFake(ConstantZ,GenX,GenZ,DisXZ,DisZ,DisX,ExpDir,name,tosave,Xnorm):
#Print Fake
    with torch.no_grad():
        FakeData = GenX(ConstantZ)
        PredFalse = DisXZ(torch.cat((DisZ(ConstantZ), DisX(FakeData)), 1))
        Z = GenZ(Xnorm)
        PredTrue = DisXZ(torch.cat((DisZ(Z), DisX(Xnorm)), 1))
        if torch.cuda.is_available():
            FakeData = FakeData.cpu()
            PredFalse = PredFalse.cpu()
        
        FakeData = FakeData.detach().numpy()
        PredFalse= PredFalse.detach().numpy()

    fig = plt.figure(figsize=(8,8))
    c = 0
    for i in range(20):
        c +=1
        #print(fd.shape)
        plt.subplot(5,5,c)
        plt.imshow(FakeData[i][0],cmap="gray",vmin=-1,vmax=1)
        #plt.imshow(FakeData[i][0],cmap="gray")
        plt.title("fDis=%.2f" % (PredFalse[i]))
        plt.axis("off")
    for i in range(5):
        c +=1
        xi = Xnorm[i]
        pi = PredTrue[i]
        if torch.cuda.is_available():
            xi = xi.cpu()
            pi = pi.cpu()
        xi = xi.detach().numpy()
        pi = pi.detach().numpy()
        plt.subplot(5,5,c)
        plt.imshow(xi[0],cmap="gray",vmin=-1,vmax=1)
        plt.title("rDis=%.2f" % (pi))
        plt.axis("off")
    fig.savefig("%s/images/GenImg/GenImg_%s_Gen_epoch_%s.png" % (ExpDir,name,tosave))
    plt.close()     
    
    
def Reconstruct(GenZ,GenX,DisX,DisZ,DisXZ,ConstantX,ExpDir,Name,tosave,SaveFile=True,Sample = 9,ImageType = "Xray" ):
    GenX.eval()
    GenZ.eval()
    DisX.eval()
    DisZ.eval()
    DisXZ.eval()

    AlphaRed = GetAlphaRedMap()
    with torch.no_grad():
    
        #Generate Latent from Real
        RealZ = GenZ(ConstantX)
        RebuildX = GenX(RealZ)
        DiffX = ConstantX - RebuildX
        
        #Have discriminator do is thing on real and fake data
        PredReal  = DisXZ(torch.cat((DisZ(RealZ), DisX(ConstantX)), 1))
        if torch.cuda.is_available():
            DiffX = DiffX.cpu()
            RebuildX = RebuildX.cpu()
            PredReal = PredReal.cpu()
            RealZ = RealZ.cpu()
            
        PredReal = PredReal.detach().numpy()
        DiffX = DiffX.detach().numpy()
        DiffX = np.power(DiffX,2)
        RebuildX = RebuildX.detach().numpy()
        RealZ = RealZ.detach().numpy()
    
    
    
    
    c = 0
    if SaveFile == True:
        fig = plt.figure(figsize=(8,8*Sample/3.0))
        for i in range(Sample):
            c+= 1
            plt.subplot(Sample,3,c)
            plt.imshow(ConstantX[i][0],cmap="gray",vmin=-1,vmax=1)
            plt.title("Init Disc=%.2f" % (PredReal[i]))
            plt.axis("off")
            c+= 1
            plt.subplot(Sample,3,c)
            plt.imshow(RebuildX[i][0],cmap="gray",vmin=-1,vmax=1)
            plt.title("Reconstruct")
            plt.axis("off")
            c+= 1
            plt.subplot(Sample,3,c)
            plt.imshow(ConstantX[i][0],cmap="gray",vmin=-1,vmax=1)
            plt.title("Rec Error = %.2f" % (np.mean(DiffX[i][0])))
            plt.imshow(DiffX[i][0],cmap=AlphaRed,vmin=0, vmax=2)
            plt.axis("off")
        #print("Saving file")
        
        if not os.path.exists("%s/images/recon/%s/" % (ExpDir,ImageType)):
            os.makedirs("%s/images/recon/%s/" % (ExpDir,ImageType))
        fig.savefig("%s/images/recon/%s/%s_%s_epoch_%s.png" % (ExpDir,ImageType,Name,ImageType,tosave))
        plt.close('all')
        
    RealZ.resize((RealZ.shape[:2]))
    return(list(np.ravel(PredReal)),[np.sqrt(np.mean(x)) for x in DiffX],list(RealZ))

def CreateFolder(wrkdir,name):
    #Default Directory with all model
    ModelDir = "./model/"
    if os.path.exists("/data/milatmp1/frappivi/ALI_model"):
        ModelDir = "/data/milatmp1/frappivi/ALI_model/"

    #If provided model dir, will used it
    if wrkdir != "NA":
        if not os.path.exists(wrkdir):
            os.makedirs(wrkdir)
        ModelDir = wrkdir
    else:
        print("No --wrkdir", wrkdir)
        
    #Create some subfolder   
    ExpDir = ModelDir+name
    if not os.path.exists(ExpDir):
        os.makedirs(ExpDir)
        os.makedirs(ExpDir+"/models")
        os.makedirs(ExpDir+"/images")
        os.makedirs(ExpDir+"/images/GenImg/")

    #This is my working dir    
    print("Wrkdir = %s" % (ExpDir))
    return(ExpDir,ModelDir)
    
    
    
def PrintTSNE(df,ToPrint = [],SaveFile="NA",MaxPlot = -1,size=20):

    fig = plt.figure(figsize=(15.5,8))
    SNScol = sns.color_palette()
    if len(ToPrint) == 0:
        ToPrint = list(df["Lab"].value_counts().index)
    for i,name in enumerate(ToPrint):
        subdf = df[df["Lab"] == name]
        xi = subdf["T-SNE1"]
        yi = subdf["T-SNE2"]
        if MaxPlot > 0:
            xi = xi[:MaxPlot]
            yi = yi[:MaxPlot]
        if len(ToPrint) != 0:
            if name not in ToPrint:
                continue
        
        plt.scatter(xi,yi,label=name,s=size,color=SNScol[i],alpha=0.7,edgecolors="none")
    #plt.legend(loc='center left',fontsize = 15,frameon=False, markerscale=2,bbox_to_anchor=(1, 0.5))
    plt.legend(fontsize = 15,frameon=False, markerscale=2)
    if SaveFile != "NA":
        fig.savefig(SaveFile)
    else:
        plt.show()
    plt.close('all')    
    
    
    
def ImageReconPrint(AllEvalData,DsetName,ToPrint = [],SaveFile="NA"):

    fig = plt.figure(figsize=(15.5,12))
    c = 1

    AlphaRed = GetAlphaRedMap()

    for i,name in enumerate(DsetName):
        if name not in AllEvalData:
            continue
        if "RecLoss" not in AllEvalData[name]:
            continue
        if len(ToPrint) != 0:
            if name not in ToPrint:
                continue
        RL = AllEvalData[name]["RecLoss"]
        Diff = AllEvalData[name]["DiffX"] 
        img = AllEvalData[name]["X"]
        RB = AllEvalData[name]["Xr"]
        SortInd = np.argsort(RL)[::-1]
        RankPrint = [0,int(len(RL)/4),int(len(RL)/2),int(len(RL)/4*3),len(RL)-1]
        for rp in RankPrint:
            NowInd = SortInd[rp]

            #Plot Init
            plt.subplot(len(DsetName),len(RankPrint)*3,c)

            plt.imshow(img[NowInd][0],cmap="gray")
            #plt.imshow(Diff[NowInd][0],cmap="gray")
            #plt.axis("off")
            plt.yticks([])
            plt.xticks([])
            if rp == RankPrint[0]:
                plt.ylabel(name)
            c += 1

            #Plt diff
            plt.subplot(len(DsetName),len(RankPrint)*3,c)
            plt.title("%.2f" % (RL[NowInd]))
            plt.imshow(Diff[NowInd][0],cmap="Reds",vmin=0, vmax=3)
            plt.yticks([])
            plt.xticks([])
            c += 1

            #Plt Rebuild
            plt.subplot(len(DsetName),len(RankPrint)*3,c)
            plt.imshow(RB[NowInd][0],cmap="gray",vmin=-1, vmax=1)
            plt.yticks([])
            plt.xticks([])
            c += 1

    if SaveFile != "NA":
        fig.savefig(SaveFile)
    else:
        plt.show()
    plt.close('all')
    
    
def ImageSortPrint(AllEvalData,DsetName,ToPrint = [],SaveFile="NA"):
    fig = plt.figure(figsize=(15.5,12))
    c = 1

    AlphaRed = GetAlphaRedMap()

    for i,name in enumerate(DsetName):
        if name not in AllEvalData:
            continue
        if "RecLoss" not in AllEvalData[name]:
            continue
        if len(ToPrint) != 0:
            if name not in ToPrint:
                continue
        RL = AllEvalData[name]["RecLoss"]
        Diff = AllEvalData[name]["DiffX"] 
        img = AllEvalData[name]["X"]
        SortInd = np.argsort(RL)[::-1]
        RankPrint = [0,1,2,3,4,int(len(RL)/4),int(len(RL)/2),int(len(RL)/4*3),len(RL)-5,len(RL)-4,len(RL)-3,len(RL)-2,len(RL)-1]
        for rp in RankPrint:
            NowInd = SortInd[rp]
            plt.subplot(len(DsetName),len(RankPrint),c)
            plt.title("%.2f" % (RL[NowInd]))
            plt.imshow(img[NowInd][0],cmap="gray")
            #plt.imshow(Diff[NowInd][0],cmap=AlphaRed,vmin=0, vmax=3)
            #plt.imshow(Diff[NowInd][0],cmap="gray")
            #plt.axis("off")
            plt.yticks([])
            plt.xticks([])
            if rp == RankPrint[0]:
                plt.ylabel(name)

            c += 1
    if SaveFile != "NA":
        fig.savefig(SaveFile)
    else:
        plt.show()
    plt.close('all')
    
    
def PrintDense(AllEvalData,DsetName,ToPrint = [],SaveFile="NA"):
    fig = plt.figure(figsize=(15.5,8))
    SNScol = sns.color_palette()
    
    for i,name in enumerate(DsetName):
        if name not in AllEvalData:
            continue
        if "RecLoss" not in AllEvalData[name]:
            continue
        if len(ToPrint) != 0:
            if name not in ToPrint:
                continue
        #Get Reconstruction loss
        TestRL = AllEvalData[name]["RecLoss"]
        RealRL = AllEvalData["ChestXray"]["RecLoss"]

        #Concat
        CatRl = RealRL + TestRL

        #Gen Label
        Lab = [1]*len(RealRL)+[0]*len(TestRL)

        #Get AUC
        fpr, tpr, thresholds = metrics.roc_curve(Lab,-np.array(CatRl), pos_label=1)
        auc = metrics.auc(fpr, tpr)
        lab =  "%s %.2f" % (name,auc)
        if name == "ChestXray":
            lab = name
        #plt.hist(TestRL,label=lab, density=True,histtype="step",bins=20,color=SNScol[i])
        sns.kdeplot(TestRL, shade=True,label=lab,color=SNScol[i])
    #plt.legend(fontsize = 30,frameon=False,prop={"family":"Sans Serif"})
    plt.legend(fontsize = 15,frameon=False)

    plt.xlabel("Reconstruction Loss",size=20)
    plt.ylabel("Density",size=20)
    plt.xticks(size=20)
    
    if SaveFile != "NA":
        fig.savefig(SaveFile)
    else:
        plt.show()
    plt.close('all')
    
    
def GetAUC(AllEvalData,metric="RecLoss"):
    tauc = dict()
    for name in AllEvalData.keys():
        if name == "ChestXray":
            continue
        if metric not in AllEvalData[name]:
            continue
        #Get Reconstruction loss
        TestRL = AllEvalData[name][metric]
        RealRL = AllEvalData["ChestXray"][metric]

        #Concat
        CatRl = RealRL + TestRL

        #Gen Label
        Lab = [1]*len(RealRL)+[0]*len(TestRL)
        mod = -1.0
        if metric == "Dis":
            mod = 1.0
        #Get AUC
        fpr, tpr, thresholds = metrics.roc_curve(Lab,mod*np.array(CatRl), pos_label=1)
        auc = metrics.auc(fpr, tpr)
        tauc[name] = auc
    return(tauc)

def GetTSNE(AllEvalData,ToPrint = []):
    AllZ = []
    lab = []
    rl = []
    imglab = []
    for name in sorted(AllEvalData.keys()):
        if "Z" not in AllEvalData[name]:continue
        if len(ToPrint) != 0:
            if name not in ToPrint:
                continue
        AllZ += AllEvalData[name]["Z"]
        lab += [name]*len(AllEvalData[name]["Z"])
        rl += list(AllEvalData[name]["RecLoss"])
        imglab += list(AllEvalData[name]["lab"])
    Y = manifold.TSNE(n_components=2).fit_transform(AllZ)
    df = pd.DataFrame([Y[:,0],Y[:,1],lab,rl,imglab]).transpose()
    df.columns = ["T-SNE1","T-SNE2","Lab","RL","ImgLab"]
    return(df)
