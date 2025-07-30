#%% IMPORTS
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
import csv
#%% CARGO ARCHIVOS
df = pd.read_excel('24h.xlsx',sheet_name='24h')
datos = df.values
encabezado = df.columns.tolist()
t = datos[:,0]
To = datos[1,1]
To_v = np.ones_like(t)*To
T_amb = datos[:,1]
Tc_A = datos[:,8]
Tc_M = datos[:,12]
Tc_B = datos[:,9]
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
Tc_B = np.reshape(Tc_B,(-1,1))
Tcal_int_M = np.reshape(Tcal_int_M,(-1,1))
Top_int_M = np.reshape(Top_int_M,(-1,1))
Tcal_int_A = np.reshape(Tcal_int_A,(-1,1))
Tcal_int_B = np.reshape(Tcal_int_B,(-1,1))
q_v = np.ones_like(t)*4
#%% PLOTS
plt.plot(t,To_v)
plt.plot(t,T_amb)
plt.show()

plt.plot(t,Tc_A,label = r'$Tc_A$')
plt.plot(t,Tc_M,label = r'$Tc_M$')
plt.plot(t,Tc_B,label = r'$Tc_B$')
plt.plot(t,Tcal_int_M,label = r'$Tcal_{int,M}$')
plt.plot(t,Top_int_M,label = r'$Top_{int,M}$')

plt.plot(t,Tcal_int_A,label = r'$Tcal_{int,A}$')
plt.plot(t,Tcal_int_B,label = r'$Tcal_{int,B}$')
plt.legend()
plt.show()

#%%CREO VECTORES COORDENADAS# en m
#X:
x_Tc_A       = np.ones_like(Tc_A)*0
x_Tc_M       = np.ones_like(Tc_M)*0
x_Tc_B       = np.ones_like(Tc_B)*0
x_Tcal_int_M = np.ones_like(Tcal_int_M)*-0.054
x_Top_int_M  = np.ones_like(Top_int_M)*0.054
x_Tcal_int_A = np.ones_like(Tcal_int_A)*-0.054
x_Tcal_int_B = np.ones_like(Tcal_int_B)*-0.054
#Y:
y_Tc_A       = np.ones_like(Tc_A)*0.02
y_Tc_M       = np.ones_like(Tc_M)*0
y_Tc_B       = np.ones_like(Tc_B)*-0.02
y_Tcal_int_M = np.ones_like(Tcal_int_M)*0
y_Top_int_M  = np.ones_like(Top_int_M)*0
y_Tcal_int_A = np.ones_like(Tcal_int_A)*0.025
y_Tcal_int_B = np.ones_like(Tcal_int_B)*-0.025


#%% ACOMODO NUEVA MATRIZ DATOS(t,x,y,To,q,T)
M_Tc_A        = np.hstack((t,x_Tc_A,y_Tc_A,To_v,q_v,Tc_A))
M_Tc_M        = np.hstack((t,x_Tc_M,y_Tc_M,To_v,q_v,Tc_M))
M_Tc_B        = np.hstack((t,x_Tc_B,y_Tc_B,To_v,q_v,Tc_B))
M_Tcal_int_M  = np.hstack((t,x_Tcal_int_M,y_Tcal_int_M,To_v,q_v,Tcal_int_M))
M_Top_int_M   = np.hstack((t,x_Top_int_M,y_Top_int_M,To_v,q_v,Top_int_M))
M_Tcal_int_A  = np.hstack((t,x_Tcal_int_A,y_Tcal_int_A,To_v,q_v,Tcal_int_A))
M_Tcal_int_B  = np.hstack((t,x_Tcal_int_B,y_Tcal_int_B,To_v,q_v,Tcal_int_B))

M_matriz = np.vstack((M_Tc_A,M_Tc_M,M_Tc_B,M_Tcal_int_M,M_Top_int_M,M_Tcal_int_A,M_Tcal_int_B))
#%% guardo nuevo archivos CSV
encab = ["t","x","y","To","q","T"]
df_N = pd.DataFrame(M_matriz, columns= encab)
df_N.to_csv("24h.csv", index=False, encoding='utf-8')
