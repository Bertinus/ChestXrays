
def CreateDataset(datadir,ExpDir,isize,N,batch_size,ModelDir,TestRatio=0.2,rseed=13,MaxSize = 1000,Testing = False,Restrict="NA"):
  
  #PreProcess folder
  PreProDir = datadir+"PreProcess/Size"+str(isize)
  ImagesInfoDF = pd.read_csv(PreProDir+"/AllImagesInfo.csv")

  #Shuffle the dataset
  np.random.seed(rseed)
  ImagesInfoDF = ImagesInfoDF.sample(frac=1.0,random_state=rseed)

  #Keep number of example
  if N > 0:
      ImagesInfoDF = ImagesInfoDF.head(N)
      
  #Split train and test
  TestSize = int(len(ImagesInfoDF)*TestRatio+0.5)
  if TestSize > MaxSize:
      TestSize = MaxSize
  
  
  
  TestDF = ImagesInfoDF.tail(TestSize)
  TrainDF = ImagesInfoDF.head(len(ImagesInfoDF)-TestSize)
  if not os.path.isfile(ExpDir+"/TrainImagesInfo.csv"):
      
      TrainDF.to_csv(ExpDir+"/TrainImagesInfo.csv")
      TestDF.to_csv(ExpDir+"/TestImagesInfo.csv")
  TrainDF = pd.read_csv(ExpDir+"/TrainImagesInfo.csv")
  TestDF = pd.read_csv(ExpDir+"/TestImagesInfo.csv")
  if Restrict != "NA":
      print("Restricting training on " + Restrict)
      print(len(TrainDF))
      TrainDF = TrainDF[TrainDF[Restrict] == 1]
      print(len(TrainDF))
  
  
  train_dataset = XrayDatasetTensor(PreProDir+"/Tensor"+str(isize)+".pt",PreProDir+"/AllImagesInfo.csv",list(TrainDF["name"]))
  test_dataset = XrayDatasetTensor(PreProDir+"/Tensor"+str(isize)+".pt",PreProDir+"/AllImagesInfo.csv",list(TestDF["name"]))
  print("Train Size = %d Test Size = %d" % (len(TrainDF),len(TestDF)))
  
  dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size,drop_last=True)
  ConstantImg = DataLoader(test_dataset, shuffle=False, batch_size=batch_size)
  
  if Testing == False:
      return(dataloader,len(TrainDF),len(TestDF),[],[])
  
  
  testing_bs = 100
  #MNIST
  MNIST_transform = transforms.Compose([transforms.Resize(isize),transforms.ToTensor()])
  MNIST_set = dset.MNIST(root=ModelDir, train=True, transform=MNIST_transform, download=True)
  MNIST_loader = DataLoader(dataset=MNIST_set,batch_size=testing_bs,shuffle=False)
  
  data_transforms = transforms.Compose([
      transforms.ToPILImage(),
      transforms.Resize([isize,isize]),
      transforms.ToTensor(),
  ])
  
  #Other Xray
  OtherXRayDir = "/media/vince/MILA/Chest_data/OtherXray/MURA-v1.1/valid/"
  if os.path.exists("/data/lisa/data/MURA-v1.1/MURA-v1.1/train/"):
      OtherXRayDir = "/data/lisa/data/MURA-v1.1/MURA-v1.1/train/"
  OtherXRay = OtherXrayDataset(OtherXRayDir, isize=isize,nrows=TestSize)
  print(len(OtherXRay))
  otherxray = DataLoader(OtherXRay, shuffle=False, batch_size=testing_bs)
  
  

  
  data_transforms = transforms.Compose([
      transforms.ToPILImage(),
      transforms.Resize(isize),
      transforms.RandomVerticalFlip(p=1.0),
      transforms.ToTensor(),
  ])
  #Add Flip
  hflip =  DataLoader(XrayDataset(datadir,TestDF, transform=data_transforms), shuffle=False, batch_size=testing_bs)
  
  data_transforms = transforms.Compose([
      transforms.ToPILImage(),
      transforms.Resize(isize),
      transforms.RandomHorizontalFlip(p=1.0),
      transforms.ToTensor(),
  ])
  #Add Flip
  vflip =  DataLoader(XrayDataset(datadir,TestDF, transform=data_transforms), shuffle=False, batch_size=testing_bs)
  randray = DataLoader(XrayDataset(datadir,TestDF, transform=data_transforms,shuffle=True), shuffle=False, batch_size=testing_bs)
  
  data_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomAffine(degrees=[-15,15],translate=(0.1,0.1),scale=(1,1.2)),
    transforms.Resize(isize),
    transforms.ToTensor(),
  ])

  RandTransLoader =  DataLoader(XrayDataset(datadir,TestDF.head(20), transform=data_transforms), shuffle=False, batch_size=batch_size)

  
  return(dataloader,len(TrainDF),len(TestDF),[ConstantImg,MNIST_loader,otherxray,hflip,vflip,randray,RandTransLoader],["XRayT","MNIST","OXray","HFlip","VFlip","Shuffle","Random"])



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
                TZdist  = TZdist[:test_size]
                break

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

        df = pd.DataFrame([AllEvalData[n]["RecLoss"],list(np.array(AllEvalData[n]["Lab"]) == "no_finding")]).transpose()
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
        
