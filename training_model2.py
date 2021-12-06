import h5py
import numpy as np
import os, sys, shutil
from PIL import Image
from pathlib import Path
import h5py
from Imageprocessor import imgprcwithpath
import csv



def train_file(path):
    #folder = path.split('/')
    folder = "Blood vessel measurement/DRIVE Retinal Dataset/version3/train"#folder[0]+'/'+folder[1]
    #print(folder)
    #input()
    return(imgprcwithpath(path, folder))

def train_directory(path, r):

    DIR = path

    files = os.listdir(path)

    for fil in files:

        fullPath = os.path.join(path, fil)

        if os.path.isdir(fullPath):

            categ, name = (Path(DIR).stem, Path(fullPath).stem)

            for f in os.listdir(fullPath):

                f = f.split('.')

                passPath = fullPath +"\\" + f[0]

                s = train_file((passPath))

                lab = f[0].split("_")

                data_p = np.concatenate((s, np.broadcast_to(np.array([int(lab[0])])[:, None, None], s.shape[:-1] + (1,))), axis = -1)
                                
                if name in (list(r.keys())):
                    
                    dataset_name = Path(passPath).stem

                    r[name].create_dataset(dataset_name, data = data_p)

def train_model():

    DIR = "Blood vessel measurement/DRIVE Retinal Dataset/version3/train"

    #DIRS = ["train\\", "test\\", "validate\\"]

    #for j in DIRS:

    #Full_path = os.path.join(DIR,'train/')

    #h5 = h5py.File('version3/data.h5', 'w') #.format(Path(Full_path).stem), 'w')

    for i in os.listdir(DIR):
        #i = i[:-4]
        if(i not in ["DMP", "DMPBW", "DMPRGB", "DMP1", "images"]):
            print(DIR+'/'+i)
            train_file(DIR+'/'+i)
        #h5.create_group(i)

    #train_directory(Full_path, h5)

    print("TRAINING DATASETS CREATED")

if __name__ == "__main__":

    sys.exit(0)
