import torch
import torch.nn as nn
import torch.nn.functional as F


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

    


