# -*- coding: utf-8 -*-
"""
Created on Wed Jul 30 11:38:33 2025

@author: Usuario
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch 
import torch.nn as nn
#import glob
#import math
os.getcwd()
#%%
path_valid = r"D:\HAAGCD\Trabajo-Final\datos preprocesados"
lista_archivos = '4.0w-08-01-25_X.xlsx'

df = pd.read_excel(os.path.join(path_valid,lista_archivos))
datos = df[:].values
encabezado = df.columns.tolist()

#%%
ind = [3,5,7,8,11,12]
plt.figure( dpi=300)
for i in range(len(ind)):
    plt.plot(datos[:,0],datos[:,ind[i]],label = encabezado[ind[i]])
plt.legend()
plt.xlabel('t[h]')
plt.ylabel('T[°C]')
plt.show()
plt.figure( dpi=300)
for i in range(len(ind)):
    plt.semilogy(datos[:,0],datos[:,ind[i]],label = encabezado[ind[i]])
plt.legend()
plt.xlabel('t[h]')
plt.ylabel('T[°C]')
plt.show()