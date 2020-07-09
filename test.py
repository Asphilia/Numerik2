#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  8 09:46:50 2020

@author: felix
"""

#!/usr/bin/env python3
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import scipy.fftpack as sfft
import plotly.express as px

# Quantisierungsmatrix aus dem Lesetext
Q1 = np.array([[8,16,24,32,40,48,56,64],
        [16,24,32,40,48,56,64,72],
        [24,32,40,48,56,64,72,80],
        [32,40,48,56,64,72,80,88],
        [40,48,56,64,72,80,88,96],
        [48,56,64,72,80,88,96,104],
        [56,64,72,80,88,96,104,112],
        [64,72,80,88,96,104,112,120]])
    
# Q aus Aufgabe 2
Q2 = np.array([[16,11,10,16,24,40,51,61],
        [12,12,14,19,26,58,60,55],
        [14,13,16,24,40,57,69,56],
        [14,17,22,29,51,87,80,62],
        [18,22,37,56,68,109,103,77],
        [24,35,55,64,81,104,113,92],
        [49,64,78,87,103,121,120,101],
        [72,92,95,98,112,100,103,99]])

def encode_quant(orig, quant):
    return (orig / quant).astype(np.int)

def decode_quant(orig, quant):
    return (orig * quant).astype(float)


def encode_dct(orig, bx, by):
    new_shape = (
        orig.shape[0] // bx * bx,
        orig.shape[1] // by * by
    )
    new = orig[
        :new_shape[0],
        :new_shape[1]
    ].reshape((
        new_shape[0] // bx,
        bx,
        new_shape[1] // by,
        by
    ))
    return sfft.dctn(new, axes=[1,3], norm='ortho')

def decode_dct(orig, bx, by):
    return sfft.idctn(orig, axes=[1,3], norm='ortho'
    ).reshape((
        orig.shape[0]*bx,
        orig.shape[2]*by
    ))
        
        
# Load image
lena = Image.open('lena.jpg')
print(lena)
plt.figure()
plt.imshow(lena, cmap = plt.get_cmap('Greys_r'))
x = np.array(lena)

quants = [1,2,4,10] # q values
blocks = [(8,8)] # block size 8x8
decs = []

for qscale in quants:
    for bx,by in blocks:
        quant = (
                ((Q1 * (qscale)))
                .clip(-100, 100)  # to prevent clipping
                .reshape((1, bx, 1, by))
            )
        
    enc = encode_dct(x,bx,by)
    encq = encode_quant(enc, quant)
    decq = decode_quant(encq, quant)
    dec = decode_dct(decq, bx, by)
    
    decs.append((dec,qscale))

for i in decs:
    print(i[0].shape)
    reconstructed = Image.fromarray(i[0].astype(np.uint8),'L')
    p = i[1]
    plt.figure()
    plt.title(f'q = {p}')
    plt.imshow(reconstructed, cmap = plt.get_cmap('Greys_r'))
    plt.show()