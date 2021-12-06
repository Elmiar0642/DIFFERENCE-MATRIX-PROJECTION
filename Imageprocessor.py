from adder import *

import numpy as np

import cupy as cp

from PIL import Image

from pathlib import Path

import sys, os

import cv2

def resizeimg(i):

    d_s = 3000

    o_s = i.size

    ratio = float(d_s)/max(o_s)

    n_s = tuple([int(x*ratio) for x in o_s])

    i = i.resize(n_s, Image.ANTIALIAS)

    n_i = Image.new("RGB", (d_s, d_s))

    n_i.paste(i, ((d_s - n_s[0])//2, 
                    (d_s - n_s[1])//2))

    return (n_i)


def imgprcwithpath(image_path, folder_name):

    #image_path = image_path.split('.')

    #image_path = image_path[0]

    img = Image.open(image_path, 'r')      #Image.open(image_path + ".jpg", 'r')         #.convert('L')

    #foldername = image_path.split("/")

    foldername = folder_name#[-3]

    print(foldername)

    image = img.resize((max(img.size), max(img.size)), Image.ANTIALIAS) #resizeimg(img)

    X = cp.array(image)

    N = X.shape[0]

    #img = Image.fromarray(np.array(X, dtype=np.uint8))

    #if(foldername not in os.listdir(os.getcwd())):
        
        #os.mkdir(foldername)

    if("DMP3" not in os.listdir(foldername)):
        
        os.mkdir("{}/{}".format(foldername, "DMP3"))
        print("made!")

    #img.save("{}/{}/{}.png".format(foldername, folder_name, Path(image_path).stem))

    L1 = 0
    #L2 = 0

    for x in range(10):   #N+1

        #print("x = {}".format(x), end="\r")

        L1 += adder(X, x, folder_name, N, foldername)
        #L2 += adder(X, x, folder_name, N, foldername)

    D = L1
    #E = L2

    #D[D>150] = 0

    #E[E>150] = 0
    
    for x in range(10):   #N+1

        print("x = {}".format(x), end="\r")

        D -= adder(cp.array(L1), x, folder_name, N, foldername)
        #E += adder(cp.array(L1), x, folder_name, N, foldername)
        #D[D<175+(x%50)] = 0

    O1 = L1 + D
    #V1 = L2 - E

    for x in range(10):   #N+1

        print("x = {}".format(x), end="\r")

        D -= adder(cp.array(O1), x, folder_name, N, foldername)
        #E += adder(cp.array(V1), x, folder_name, N, foldername)
        #D[D<175+(x%50)] = 0

    O2 = O1 + D
    #V2 = V1 - E

    for x in range(10):   #N+1

        print("x = {}".format(x), end="\r")

        D -= adder(cp.array(O2), x, folder_name, N, foldername)
        #E += adder(cp.array(V2), x, folder_name, N, foldername)
    
    D = O2 + D 

    D = ~D
    
    D = cp.asnumpy(X) - D

    spot = D

    #img2 = Image.fromarray(np.array(D, dtype=np.uint8))

    #img2.save("{}/{}spot.png".format(foldername+'/DMP2', Path(image_path).stem))

    D = ~D 

    D[D>150] = 0 

    veins = D

    #img3 = Image.fromarray(np.array(D, dtype=np.uint8))

    #img3.save("{}/{}veins.png".format(foldername+'/DMP2', Path(image_path).stem))

    segmented = spot - veins

    segmented = cv2.bitwise_and(segmented, segmented)

    X = cp.asnumpy(X)

    segmented = cv2.bitwise_and(X, segmented)

    img4 = Image.fromarray(np.array(segmented, dtype=np.uint8))

    img4.save("{}/{}spot.png".format(foldername+'/DMP3', Path(image_path).stem))
    
    print("DONE DMP")

    return(np.array(D, dtype=np.uint8))
    
    del(L1, D, O1, O2)
    

def imgprc(single_image, folder_name):

    image = Image.open(single_image + ".png", 'r')         #.convert('L')

    X = np.array(image)

    N = X.shape[0]

    img = Image.fromarray(np.array(X, dtype=np.uint8))

    foldername = single_image.split("\\")

    foldername = foldername[-3]

    os.mkdir("{}/{}".format(foldername, folder_name))

    img.save("{}/{}/{}.png".format(foldername, folder_name, folder_name))

    D = 0

    for x in range(8):   #N+1

        D += adder(X, x, folder_name, N, foldername)
        
        img2 = Image.fromarray(np.array(D, dtype=np.uint8))

        img2.save("{}/{}/{}/{}_Filtered{}.png".format(os.getcwd(), foldername, folder_name, folder_name, x))

    img2 = Image.fromarray(np.array(D, dtype=np.uint8))

    img2.save("{}/{}/{}/{}_DMP.png".format(os.getcwd(), foldername, folder_name, folder_name))

def dataset_getter(Datapath, foldername):

    imgprc(Datapath, foldername)

if __name__ == "__main__":

    sys.exit(0)