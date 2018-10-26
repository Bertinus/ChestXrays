import torch
import torch.nn as nn
import torch.nn.functional as F



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
            self.main.add_module("ConvT_"+str(i), torch.nn.ConvTranspose2d(lastdepth,nnc,kernel_size,stride,padding,output_pading,bias=False))
            if i != len(KS) - 1:
                self.main.add_module("BN_"+str(i), torch.nn.BatchNorm2d(nnc))
                self.main.add_module("Relu_"+str(i), torch.nn.ReLU(True))
            
            OldDim = (OldDim-1)*stride+kernel_size - 2*padding + output_pading
            lastdepth = nnc
            #print("I=%d K=%d ST=%d Size=%d" % (i,kernel_size,stride,OldDim))
            
            
        #self.main.add_module("Sigmoid",nn.Sigmoid())
        self.main.add_module("Tanh",nn.Tanh())
       

    def forward(self, input):
        return self.main(input)



class Encoder(nn.Module):
    def __init__(self,KS,ST,DP,LS):
        super(Encoder, self).__init__()
        
        
        #Build ConvTranspose layer
        self.main = torch.nn.Sequential()
        
        
        #torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0
        
        self.main = torch.nn.Sequential()
        lastdepth = 1
        nnc = 1
        for i in range(len(KS)):
            
            kernel_size = KS[i]
            stride = ST[i]
            nnc = DP[i]
            
            padding = 0
            output_pading = 0
            
            self.main.add_module("ConvT_"+str(i), 
                                 torch.nn.Conv2d(in_channels=lastdepth,out_channels=nnc,
                                                 kernel_size=kernel_size,stride=stride,bias=False))
           
            if i != len(KS) - 1:
                self.main.add_module("BN_"+str(i), torch.nn.BatchNorm2d(nnc))
                self.main.add_module("LRelu_"+str(i), torch.nn.LeakyReLU(0.1, inplace=True))
            
            lastdepth = nnc
    
       

    def forward(self, input):
        return self.main(input)














class DiscriminatorX(nn.Module):
    def __init__(self,KS,ST,DP):
        super(DiscriminatorX, self).__init__()
        
        
        #Build ConvTranspose layer
        self.main = torch.nn.Sequential()
        
        
        #torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0
        
        self.main = torch.nn.Sequential()
        lastdepth = 1
        nnc = 1
        dp =0.5
        for i in range(len(KS)):
            
            kernel_size = KS[i]
            stride = ST[i]
            nnc = DP[i]
            
            padding = 0
            output_pading = 0
            
            self.main.add_module("ConvT_"+str(i), 
                                 torch.nn.Conv2d(in_channels=lastdepth,out_channels=nnc,
                                                 kernel_size=kernel_size,stride=stride,bias=False))
           
            if i != len(KS) - 1:
                self.main.add_module("DropOut_"+str(i), torch.nn.Dropout(dp))
                self.main.add_module("LRelu_"+str(i), torch.nn.LeakyReLU(0.1, inplace=True))
            
            lastdepth = nnc
            dp = 0.2
       

    def forward(self, input):
        return self.main(input)


    
    
#torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)

class DiscriminatorZ(nn.Module):
    def __init__(self,KS,ST,DP,LS):
        super(DiscriminatorZ, self).__init__()
        
        
        #Build ConvTranspose layer
        self.main = torch.nn.Sequential()
        
        
        #torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0
        
        self.main = torch.nn.Sequential()
        lastdepth = LS
        nnc = 1
        dp = 0.5
        for i in range(len(KS)):
            
            kernel_size = KS[i]
            stride = ST[i]
            nnc = DP[i]
            
            padding = 0
            output_pading = 0
            
            self.main.add_module("ConvT_"+str(i), 
                                 torch.nn.Conv2d(in_channels=lastdepth,out_channels=nnc,
                                                 kernel_size=kernel_size,stride=stride,bias=False))
           
            if i != len(KS) - 1:
                self.main.add_module("DropOut_"+str(i), torch.nn.Dropout(dp))
                self.main.add_module("LRelu_"+str(i), torch.nn.LeakyReLU(0.1, inplace=True))
            
            lastdepth = nnc
            dp = 0.2
           
       

    def forward(self, input):
        return self.main(input)


class DiscriminatorXZ(nn.Module):
    def __init__(self,KS,ST,DP):
        super(DiscriminatorXZ, self).__init__()
        
        
        #Build ConvTranspose layer
        self.main = torch.nn.Sequential()
        
        
        #torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0
        
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
            
            self.main.add_module("ConvT_"+str(i), 
                                 torch.nn.Conv2d(in_channels=lastdepth,out_channels=nnc,
                                                 kernel_size=kernel_size,stride=stride,bias=False))
            
            
            if i != len(KS) - 1:
                self.main.add_module("DropOut_"+str(i), torch.nn.Dropout(dp))
                self.main.add_module("LRelu_"+str(i), torch.nn.LeakyReLU(0.1, inplace=True))
                #if nnc > 10:
                #    self.main.add_module("Maxout_"+str(i),Maxout(2))

            lastdepth = nnc
            dp = 0.2
        self.main.add_module("Sigmoid", torch.nn.Sigmoid())
           
       

    def forward(self, input):
        return self.main(input)

    


