"""
Creador de base de datos
se procece de a un archivos a la vez

"""
#%% IMPORTS
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import numpy as np
#import torch
#import matplotlib.pyplot as plt
import pandas as pd
import csv
import glob
os.getcwd()
#%%
path = r"D:\HAAGCD\Trabajo-Final\datos preprocesados"
lista_archivos = glob.glob("*.xlsx")
#%%
DATOS = np.empty((0,6))
for i in range(len(lista_archivos)):
    path_arch = os.path.join(path,lista_archivos[i])
    print(lista_archivos[i])
    q=float(lista_archivos[i][0:3])
    df = pd.read_excel(path_arch)
    datos = df.values
    #encabezado = df.columns.tolist()
    
    t = datos[:,0]
    To = datos[1,1]
    To_v = np.ones_like(t)*To
    T_amb = datos[:,1]
    Tc_A = datos[:,8]
    Tc_M = datos[:,12]
    Tcal_int_M = datos[:,3]
    Top_int_M = datos[:,11]
    Tcal_int_A = datos[:,7]
    Tcal_int_B = datos[:,5]
    
    #RESHAPE A COLUMNAS
    t = np.reshape(t,(-1,1))
    To_v = np.reshape(To_v,(-1,1))
    T_amb = np.reshape(T_amb,(-1,1))
    Tc_A = np.reshape(Tc_A,(-1,1))
    Tc_M = np.reshape(Tc_M,(-1,1))
    Tcal_int_M = np.reshape(Tcal_int_M,(-1,1))
    Top_int_M = np.reshape(Top_int_M,(-1,1))
    Tcal_int_A = np.reshape(Tcal_int_A,(-1,1))
    Tcal_int_B = np.reshape(Tcal_int_B,(-1,1))
    q_v = np.ones_like(t)*q
    
    #X:
    x_Tc_A       = np.ones_like(Tc_A)*0
    x_Tc_M       = np.ones_like(Tc_M)*0
    x_Tcal_int_M = np.ones_like(Tcal_int_M)*-0.054
    x_Top_int_M  = np.ones_like(Top_int_M)*0.054
    x_Tcal_int_A = np.ones_like(Tcal_int_A)*-0.054
    x_Tcal_int_B = np.ones_like(Tcal_int_B)*-0.054
    #Y:
    y_Tc_A       = np.ones_like(Tc_A)*0.02
    y_Tc_M       = np.ones_like(Tc_M)*0
    y_Tcal_int_M = np.ones_like(Tcal_int_M)*0
    y_Top_int_M  = np.ones_like(Top_int_M)*0
    y_Tcal_int_A = np.ones_like(Tcal_int_A)*0.025
    y_Tcal_int_B = np.ones_like(Tcal_int_B)*-0.025

    M_Tc_A        = np.hstack((t,x_Tc_A,y_Tc_A,To_v,q_v,Tc_A))
    M_Tc_M        = np.hstack((t,x_Tc_M,y_Tc_M,To_v,q_v,Tc_M))
    M_Tcal_int_M  = np.hstack((t,x_Tcal_int_M,y_Tcal_int_M,To_v,q_v,Tcal_int_M))
    M_Top_int_M   = np.hstack((t,x_Top_int_M,y_Top_int_M,To_v,q_v,Top_int_M))
    M_Tcal_int_A  = np.hstack((t,x_Tcal_int_A,y_Tcal_int_A,To_v,q_v,Tcal_int_A))
    M_Tcal_int_B  = np.hstack((t,x_Tcal_int_B,y_Tcal_int_B,To_v,q_v,Tcal_int_B))
    
    M_matriz = np.vstack((M_Tc_A,M_Tc_M,M_Tcal_int_M,M_Top_int_M,M_Tcal_int_A,M_Tcal_int_B))
    if lista_archivos[i][14]=='O':
        Tc_B = datos[:,9]
        Tc_B = np.reshape(Tc_B,(-1,1))
        x_Tc_B = np.ones_like(Tc_B)*0
        y_Tc_B = np.ones_like(Tc_B)*-0.02
        M_Tc_B = np.hstack((t,x_Tc_B,y_Tc_B,To_v,q_v,Tc_B))
        M_matriz = np.vstack((M_matriz,M_Tc_B))
    DATOS = np.vstack((DATOS,M_matriz))
    del df,datos
    
#%% guardo nuevo archivos CSV
encab = ["t","x","y","To","q","T"]
df_N = pd.DataFrame(DATOS, columns= encab)
df_N.to_csv("DATOS3.csv", index=False, encoding='utf-8')
