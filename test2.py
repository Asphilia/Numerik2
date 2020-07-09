#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  8 19:43:40 2020

@author: felix
"""

import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import scipy.fftpack as sfft
import plotly.express as px


Q1 = np.array([[8,16,24,32,40,48,56,64],
        [16,24,32,40,48,56,64,72],
        [24,32,40,48,56,64,72,80],
        [32,40,48,56,64,72,80,88],
        [40,48,56,64,72,80,88,96],
        [48,56,64,72,80,88,96,104],
        [56,64,72,80,88,96,104,112],
        [64,72,80,88,96,104,112,120]])

    
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
testpic = Image.open('TestBild.jpg')
print(testpic)
plt.figure()
plt.imshow(testpic, cmap = plt.get_cmap('Greys_r'))
picAr = np.array(testpic)

def quant1(V8,p):
    
    new_shape = (
        V8.shape[0] // 8 * 8,
        V8.shape[1] // 8 * 8
    )
    new = V8[
        :new_shape[0],
        :new_shape[1]
    ].reshape((
        new_shape[0] // 8,
        8,
        new_shape[1] // 8,
        8
    ))
    enc = sfft.dct(new, norm='ortho')
    '''
    quant = (
                (np.ones((8,8))
                 .clip(-100,100)
                 .reshape((1,8,1,8)))
                )
    '''
    quant = (
                ((Q1 * (p)))
                .clip(-100, 100)  # to prevent clipping
                .reshape((1,8,1,8))
            )
    
    encq = encode_quant(enc, quant)
    decq = decode_quant(encq, quant)
    r = sfft.idct(decq,norm = 'ortho')
    dec =r.reshape((
        decq.shape[0]*8,
        decq.shape[2]*8
    ))

    reconstructed = Image.fromarray(dec.astype(np.uint8),'L')
    plt.figure()
    plt.imshow(reconstructed, cmap = plt.get_cmap('Greys_r'))
    plt.show()


quant1(picAr,1)
    