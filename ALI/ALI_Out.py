import torch
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.colors import ListedColormap
import os



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
