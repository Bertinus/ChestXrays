import numpy as np
import pandas as pd
from scipy import misc
import os

"""
File that allows to remove lines in a dataset csv if the corresponding image is missing
"""


csvpath = "/network/home/bertinpa/Documents/ChestXrays/PADCHEST_chest_x_ray_images_labels_160K_01.02.19.csv"
datadir = "/network/data1/academictorrents-datastore/PADCHEST_SJ/image"

Data = pd.read_csv(csvpath, low_memory=False)
L = []

print(Data.shape)

for idx in len(Data):
    if not os.path.isfile(os.path.join(datadir, Data['ImageID'][idx])):
        L.append(idx)

print(Data.drop(L).shape)
