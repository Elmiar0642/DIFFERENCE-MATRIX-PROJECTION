from identitymatrixgenerator import *

import numpy as np

import cupy as cp 

import sys

def pixeldifmap(b, N):

    h_H_l = cp.array(pixel_len(b, N), dtype = cp.uint8)

    v_H_l = cp.array(cp.transpose(h_H_l), dtype = cp.uint8)

    I = cp.identity(N)

    h_P_l = I - h_H_l

    h_P_l = cp.array(h_P_l, dtype = cp.uint8)

    h_P_l = h_P_l.reshape((h_P_l.shape[0], h_P_l.shape[1], 1))

    v_P_l = I - v_H_l

    v_P_l = cp.array(v_P_l, dtype = cp.uint8)

    v_P_l = v_P_l.reshape((v_P_l.shape[0], v_P_l.shape[1], 1))

    return([h_P_l, v_P_l])

    del(h_H_l, v_H_l, I, h_P_l, v_P_l)

if __name__ == "__main__":

    sys.exit(0)
