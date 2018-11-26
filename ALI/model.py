import torch
import torch.nn as nn
import torch.nn.functional as F
import glob as glob
import os
import pickle

def GenModel(size,LS,CP,ExpDir,name,ColorsNumber=1):
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
    
    if size == 64:
        #Generator param
        EncKernel = [2,7,5,7,4,1]
        EncStride = [1,2,2,2,1,1]
        EncDepth = [64,128,256,512,512,LS]

        #Generator param
        GenKernel = [4,7,5,7,2,1]
        GenStride = [1,2,2,2,1,1]
        GenDepth = [256,128,64,32,32,32,ColorsNumber]

        #Discriminator X param
        DxKernel = [2,7,5,7,4]
        DxStride = [1,2,2,2,1]
        DxDepth = [64,128,256,256,512]

        #Discriminator Z param
        DzKernel = [1,1]
        DzStride = [1,1]
        DzDepth = [512,512]

        #Concat Discriminator param
        DxzKernel = [1,1,1]
        DxzStride = [1,1,1]
        DxzDepth = [2048,2048,1]
    
    
    #Create Model

    DisX = DiscriminatorX(KS=DxKernel,ST=DxStride,DP=DxDepth)
    DisZ = DiscriminatorZ(KS=DzKernel,ST=DzStride,DP=DzDepth,LS=LS)
    DisXZ = DiscriminatorXZ(KS=DxzKernel,ST=DxzStride,DP=DxzDepth)

    GenZ = Encoder(KS=EncKernel,ST=EncStride,DP=EncDepth,LS=LS)
    GenX = Generator(latent_size=LS,KS=GenKernel,ST=GenStride,DP=GenDepth)
    
    #Check checkpoint to use
    if CP == -2:
        #Find latest
        MaxCk = 0
        for fck in glob.glob('{0}/models/*_DisXZ_It_*.pth'.format(ExpDir)):
            #print(fck)
            nck = fck.split("_")[-1].split(".")[0]
            if int(nck) > MaxCk:MaxCk = int(nck)
        CP = MaxCk
        print("I found this last checkpoint %d" % (CP))
    Diss = dict()
    AllAUCs = dict()
    #Check if checkpoint exist
    if os.path.isfile('{0}/models/{1}_DisXZ_It_{2}.pth'.format(ExpDir,name, CP)):
        #print("Checkpoint %d exist, will load param and start training from there" % (CP))
        DisX.load_state_dict(torch.load('{0}/models/{1}_DisX_It_{2}.pth'.format(ExpDir,name, CP),map_location={'cuda:0': 'cpu'}))
        DisZ.load_state_dict(torch.load('{0}/models/{1}_DisZ_It_{2}.pth'.format(ExpDir,name, CP),map_location={'cuda:0': 'cpu'}))
        DisXZ.load_state_dict(torch.load('{0}/models/{1}_DisXZ_It_{2}.pth'.format(ExpDir,name, CP),map_location={'cuda:0': 'cpu'}))
        
        GenZ.load_state_dict(torch.load('{0}/models/{1}_GenZ_It_{2}.pth'.format(ExpDir,name, CP),map_location={'cuda:0': 'cpu'}))
        GenX.load_state_dict(torch.load('{0}/models/{1}_GenX_It_{2}.pth'.format(ExpDir,name, CP),map_location={'cuda:0': 'cpu'}))
        Diss = pickle.load(open('{0}/models/{1}_Loss_It_{2}.pth'.format(ExpDir,name, CP),"rb"))
        
        if os.path.isfile('{0}/models/{1}_AUCs_It_{2}.pth'.format(ExpDir,name, CP)):
            AllAUCs = pickle.load(open('{0}/models/{1}_AUCs_It_{2}.pth'.format(ExpDir,name, CP),"rb"))
        #print("Done loading")

    if torch.cuda.is_available():
        #print("cuda is available")
        DisX = DisX.cuda()
        DisZ = DisZ.cuda()
        DisXZ = DisXZ.cuda()

        GenZ = GenZ.cuda()
        GenX = GenX.cuda()                                    

    
    
    
    
    return(DisX,DisZ,DisXZ,GenZ,GenX,CP,Diss,AllAUCs)
    
    

#Generator model (Gx(z))
class Generator(nn.Module):
    def __init__(self, latent_size=32,output_shape=224,nc=1,KS= [4,221],ST = [1,1],DP=[1,1]):
        self.latent_size = latent_size
        self.output_shape = output_shape
        self.nc = nc
        
        super(Generator, self).__init__()
        
        #Build ConvTranspose layer
        self.main = torch.nn.Sequential()
        lastdepth = self.latent_size
        OldDim = 1
        for i in range(len(KS)):
            #Depth
            nnc = DP[i]
            #Kernel Size
            kernel_size = KS[i]
            #Stride
            stride = ST[i]
            
            #Default value
            padding = 0
            output_pading = 0
            
            if i == len(KS)-1:
                nnc = 1
                
            #Add ConvTranspose
            self.main.add_module("ConvT_"+str(i), torch.nn.ConvTranspose2d(lastdepth,nnc,kernel_size,stride,padding,output_pading,bias=False))
            
            #Some regurlarisation
            if i != len(KS) - 1:
                self.main.add_module("Relu_"+str(i), torch.nn.ReLU(True))
                self.main.add_module("BN_"+str(i), torch.nn.BatchNorm2d(nnc))            
            #OldDimension (for information)
            OldDim = (OldDim-1)*stride+kernel_size - 2*padding + output_pading
            
            #Last depth (to pass to next ConvT layer)
            lastdepth = nnc
            #print("I=%d K=%d ST=%d Size=%d" % (i,kernel_size,stride,OldDim))
            
            
        #self.main.add_module("Sigmoid",nn.Sigmoid())
        self.main.add_module("Tanh",nn.Tanh()) #Apparently Tanh is better than Sigmoid
       
    def forward(self, input):
        return self.main(input)


#Image Encoder network to latent space (Gz(x))
class Encoder(nn.Module):
    def __init__(self,KS,ST,DP,LS):
        super(Encoder, self).__init__()
        
        
        #Sequential model        
        self.main = torch.nn.Sequential()
        lastdepth = 1 #This is the number of color (1)
        nnc = 1
        for i in range(len(KS)):
            
            #Kernel, Stride and Depth from param
            kernel_size = KS[i]
            stride = ST[i]
            nnc = DP[i]
            
            #No padding!
            padding = 0
            output_pading = 0
            
            #Conv layer
            self.main.add_module("Conv_"+str(i), 
                                 torch.nn.Conv2d(in_channels=lastdepth,out_channels=nnc,
                                                 kernel_size=kernel_size,stride=stride,bias=False))
           
           #Some regul
            if i != len(KS) - 1:
                self.main.add_module("LRelu_"+str(i), torch.nn.LeakyReLU(0.1, inplace=True))
                self.main.add_module("BN_"+str(i), torch.nn.BatchNorm2d(nnc))
            lastdepth = nnc

    def forward(self, input):
        return self.main(input)

#Discriminator X (Take an image and discriminate it) Dx(x)
class DiscriminatorX(nn.Module):
    def __init__(self,KS,ST,DP):
        super(DiscriminatorX, self).__init__()
        
        self.main = torch.nn.Sequential()
        lastdepth = 1
        nnc = 1
        dp =0.5 #Dropout rate is 0.5 for first
        for i in range(len(KS)):
        
            #Kernel, Stride and Depth from param
            kernel_size = KS[i]
            stride = ST[i]
            nnc = DP[i]
            
            #No padding
            padding = 0
            output_pading = 0
            
            self.main.add_module("Conv_"+str(i), 
                                 torch.nn.Conv2d(in_channels=lastdepth,out_channels=nnc,
                                                 kernel_size=kernel_size,stride=stride,bias=False))
            #Some regularization
            if i != len(KS) - 1:
                self.main.add_module("LRelu_"+str(i), torch.nn.LeakyReLU(0.1, inplace=True))
                self.main.add_module("DropOut_"+str(i), torch.nn.Dropout(dp))
            
            lastdepth = nnc
            dp = 0.2 #New dropout rate
       

    def forward(self, input):
        return self.main(input)


    
    
#Discriminator for Latent Space (Dz(z))
class DiscriminatorZ(nn.Module):
    def __init__(self,KS,ST,DP,LS):
        super(DiscriminatorZ, self).__init__()
        
        self.main = torch.nn.Sequential()
        lastdepth = LS
        nnc = 1
        dp = 0.5
        for i in range(len(KS)):
            
            #Kernel, Stride and Depth from param
            kernel_size = KS[i]
            stride = ST[i]
            nnc = DP[i]
            
            #No padding!
            padding = 0
            output_pading = 0
            
            #Conv
            self.main.add_module("Conv_"+str(i), 
                                 torch.nn.Conv2d(in_channels=lastdepth,out_channels=nnc,
                                                 kernel_size=kernel_size,stride=stride,bias=False))
           
            if i != len(KS) - 1:
                self.main.add_module("LRelu_"+str(i), torch.nn.LeakyReLU(0.1, inplace=True))
                self.main.add_module("DropOut_"+str(i), torch.nn.Dropout(dp))            
            lastdepth = nnc
            dp = 0.2

    def forward(self, input):
        return self.main(input)


class DiscriminatorXZ(nn.Module):
    def __init__(self,KS,ST,DP):
        super(DiscriminatorXZ, self).__init__()
        
        self.main = torch.nn.Sequential()
        lastdepth = 1024
        nnc = 1
        dp = 0.5
        for i in range(len(KS)):
            
            kernel_size = KS[i]
            stride = ST[i]
            nnc = DP[i]
            
            padding = 0
            output_pading = 0
            
            self.main.add_module("Conv_"+str(i), 
                                 torch.nn.Conv2d(in_channels=lastdepth,out_channels=nnc,
                                                 kernel_size=kernel_size,stride=stride,bias=False))
            
            
            if i != len(KS) - 1:
                self.main.add_module("LRelu_"+str(i), torch.nn.LeakyReLU(0.1, inplace=True))
                self.main.add_module("DropOut_"+str(i), torch.nn.Dropout(dp))

            lastdepth = nnc
            dp = 0.2
        self.main.add_module("Sigmoid", torch.nn.Sigmoid())
           
       

    def forward(self, input):
        return self.main(input)

    


