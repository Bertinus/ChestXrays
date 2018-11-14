import os


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

    #This is my working dir    
    print("Wrkdir = %s" % (ExpDir))
    return(ExpDir)
