"""
RED NEURONAL--> Validacion
MLP
5 entradas - 1 salida
t,x_i,y_i,To,q ---> T_i
Autor: Matias Sosa
"""
#%% Imports
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
#%%RED NEURONAL
#entrada: t,x,y,To,q  salida: T
#defino la misma clase
class Mired(nn.Module):
   
    def __init__(self, n_in, n_hidd, n_out):
      
        super().__init__()
        self.Lin_IN = nn.Linear(n_in, n_hidd) 
        self.act1 =  torch.nn.Tanh()
        self.Lin2 = nn.Linear(n_hidd, n_hidd)
        self.act2 = torch.nn.Tanh()
        self.Lin3 = nn.Linear(n_hidd, n_hidd)
        self.act3 = torch.nn.Tanh()
        self.Lin4 = nn.Linear(n_hidd, n_hidd)
        self.act4 = torch.nn.Tanh()
        self.Lin_OUT = nn.Linear(n_hidd, n_out)
       
    def forward(self, x):
        x=self.Lin_IN(x)
        x=self.act1(x)
        x=self.Lin2(x)
        x=self.act2(x)
        x=self.Lin3(x)
        x=self.act3(x)
        x=self.Lin4(x)
        x=self.act4(x)
        salida = self.Lin_OUT(x)
              
        return salida
# Creacion de la red
#def de los mismos tamaños
input_size = 5
output_size = 1
n_hidden_size = 64
#instancio la red
model = Mired(input_size,n_hidden_size, output_size)
#declaro la metrica y el optimizador
crit = nn.HuberLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
#%% Cargo los parametros
SD = torch.load("mired-4capas-64hidden_14mil-epocas.pth")
#SD.keys()
model.load_state_dict(SD)
#%% LISTO, YA TENGO MI MODELO RECONSTRUIDO
#%% Cargo dato reservado para validacion
path_valid = r"D:\HAAGCD\Trabajo-Final\datos preprocesados"
lista_archivos = '4.0w-08-01-25_X.xlsx'

q=float(lista_archivos[0:3])
df = pd.read_excel(os.path.join(path_valid,lista_archivos))
datos = df[:-10].values
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
if lista_archivos[14]=='O':
    Tc_B = datos[:,9]
    Tc_B = np.reshape(Tc_B,(-1,1))
    x_Tc_B = np.ones_like(Tc_B)*0
    y_Tc_B = np.ones_like(Tc_B)*-0.02
    M_Tc_B = np.hstack((t,x_Tc_B,y_Tc_B,To_v,q_v,Tc_B))
    M_matriz = np.vstack((M_matriz,M_Tc_B))
loss_modelo = []
#%%% Tc_A
X_Tc_A = torch.FloatTensor(M_Tc_A[:,0:5]).requires_grad_().view(-1,5)
Y_Tc_A = torch.FloatTensor(M_Tc_A[:,5]).requires_grad_().view(-1,1)

model.eval() # Poner el modelo en modo evaluacion
with torch.no_grad(): # Desactiva el calculo de gradientes
    Y_Tc_A_pred = model(X_Tc_A) #Valuo modelo en el test
    test_loss = crit(Y_Tc_A_pred, Y_Tc_A) #calculo metrica

    print(f'Pérdida en el conjunto de prueba: {test_loss.item():.4f}')
    loss_modelo.append(test_loss.item())
    
#%%% Tc_M 
X_Tc_M = torch.FloatTensor(M_Tc_M[:,0:5]).requires_grad_().view(-1,5)
Y_Tc_M = torch.FloatTensor(M_Tc_M[:,5]).requires_grad_().view(-1,1)

model.eval() # Poner el modelo en modo evaluacion
with torch.no_grad(): # Desactiva el calculo de gradientes
    Y_Tc_M_pred = model(X_Tc_M) #Valuo modelo en el test
    test_loss = crit(Y_Tc_M_pred, Y_Tc_M) #calculo metrica

    print(f'Pérdida en el conjunto de prueba: {test_loss.item():.4f}')
    loss_modelo.append(test_loss.item())
#%%% Tc_B 
if lista_archivos[14]=='O':
    X_Tc_B = torch.FloatTensor(M_Tc_B[:,0:5]).requires_grad_().view(-1,5)
    Y_Tc_B = torch.FloatTensor(M_Tc_B[:,5]).requires_grad_().view(-1,1)
    
    model.eval() # Poner el modelo en modo evaluacion
    with torch.no_grad(): # Desactiva el calculo de gradientes
        Y_Tc_B_pred = model(X_Tc_B) #Valuo modelo en el test
        test_loss = crit(Y_Tc_B_pred, Y_Tc_B) #calculo metrica
    
        print(f'Pérdida en el conjunto de prueba: {test_loss.item():.4f}')
        loss_modelo.append(test_loss.item())
else:
    print('no hay medida de Tc_B')
#%%% Tcal_int_M
X_Tcal_int_M = torch.FloatTensor(M_Tcal_int_M[:,0:5]).requires_grad_().view(-1,5)
Y_Tcal_int_M= torch.FloatTensor(M_Tcal_int_M[:,5]).requires_grad_().view(-1,1)

model.eval() # Poner el modelo en modo evaluacion
with torch.no_grad(): # Desactiva el calculo de gradientes
    Y_Tcal_int_M_pred = model(X_Tcal_int_M) #Valuo modelo en el test
    test_loss = crit(Y_Tcal_int_M_pred, Y_Tcal_int_M) #calculo metrica

    print(f'Pérdida en el conjunto de prueba: {test_loss.item():.4f}')
    loss_modelo.append(test_loss.item())
#%%% Top_int_M
X_Top_int_M = torch.FloatTensor(M_Top_int_M[:,0:5]).requires_grad_().view(-1,5)
Y_Top_int_M= torch.FloatTensor(M_Top_int_M[:,5]).requires_grad_().view(-1,1)

model.eval() # Poner el modelo en modo evaluacion
with torch.no_grad(): # Desactiva el calculo de gradientes
    Y_Top_int_M_pred = model(X_Top_int_M) #Valuo modelo en el test
    test_loss = crit(Y_Top_int_M_pred, Y_Top_int_M) #calculo metrica

    print(f'Pérdida en el conjunto de prueba: {test_loss.item():.4f}')
    loss_modelo.append(test_loss.item())
#%%% Tcal_int_A
X_Tcal_int_A = torch.FloatTensor(M_Tcal_int_A[:,0:5]).requires_grad_().view(-1,5)
Y_Tcal_int_A = torch.FloatTensor(M_Tcal_int_A[:,5]).requires_grad_().view(-1,1)

model.eval() # Poner el modelo en modo evaluacion
with torch.no_grad(): # Desactiva el calculo de gradientes
    Y_Tcal_int_A_pred = model(X_Tcal_int_A) #Valuo modelo en el test
    test_loss = crit(Y_Tcal_int_A_pred, Y_Tcal_int_A) #calculo metrica

    print(f'Pérdida en el conjunto de prueba: {test_loss.item():.4f}')
    loss_modelo.append(test_loss.item())

#%%% Tcal_int_B
X_Tcal_int_B = torch.FloatTensor(M_Tcal_int_B[:,0:5]).requires_grad_().view(-1,5)
Y_Tcal_int_B= torch.FloatTensor(M_Tcal_int_B[:,5]).requires_grad_().view(-1,1)

model.eval() # Poner el modelo en modo evaluacion
with torch.no_grad(): # Desactiva el calculo de gradientes
    Y_Tcal_int_B_pred = model(X_Tcal_int_B) #Valuo modelo en el test
    test_loss = crit(Y_Tcal_int_B_pred, Y_Tcal_int_B) #calculo metrica

    print(f'Pérdida en el conjunto de prueba: {test_loss.item():.4f}')
    loss_modelo.append(test_loss.item())
#%% ploteo de predicciones y medidas
Y_Tc_A_pred_np = Y_Tc_A_pred.squeeze().numpy()
Y_Tc_M_pred_np = Y_Tc_M_pred.squeeze().numpy()
Y_Tcal_int_M_pred_np = Y_Tcal_int_M_pred.squeeze().numpy()
Y_Top_int_M_pred_np = Y_Top_int_M_pred.squeeze().numpy()
Y_Tcal_int_A_pred_np = Y_Tcal_int_A_pred.squeeze().numpy()
Y_Tcal_int_B_pred_np = Y_Tcal_int_B_pred.squeeze().numpy()

plt.figure( dpi=300)

plt.plot(t,Y_Tc_A_pred_np,label='Tc_A_pred')
plt.plot(t,M_Tc_A[:,5],label='Tc_A_dato')

plt.plot(t,Y_Tc_M_pred_np,label='Tc_M_pred')
plt.plot(t,M_Tc_M[:,5],label='Tc_M_dato')

plt.plot(t,Y_Tcal_int_M_pred_np,label='Tcal_int_M_pred')
plt.plot(t,M_Tcal_int_M[:,5],label='Tcal_int_M_dato')

plt.plot(t,Y_Top_int_M_pred_np,label='Top_int_M_pred')
plt.plot(t,M_Top_int_M[:,5],label='Top_int_M_dato')

plt.ylim([30,34])
plt.xlim([2,14])
plt.legend()
plt.xlabel('t[h]')
plt.ylabel('T[°C]')
plt.show()

#%% Def funcion exponencial

from scipy.optimize import curve_fit
t_data = t

#Def fun exponencial para el ajuste
def exponential_decay(t, y0, A1, t1):
    """
    Función de decaimiento exponencial: T(t) = y0 + A1 * exp(-t/t1)
    """
    return y0 + A1 * np.exp(-t / t1)


#%% Ajuste exponencial
initial_guess = np.array([30.,-6., 3.])
MAT      = np.array([Tc_A,Tc_M,Tcal_int_M,Top_int_M,Tcal_int_A,Tcal_int_B])
MAT_pred = np.array([Y_Tc_A_pred_np,Y_Tc_M_pred_np,Y_Tcal_int_M_pred_np,Y_Top_int_M_pred_np,Y_Tcal_int_A_pred_np,Y_Tcal_int_B_pred_np])
param = []
MAT_nom = ['Tc_A','Tc_M','Tcal_int_M','Top_int_M','Tcal_int_A','Tcal_int_B']
for i in range(MAT.shape[0]):
    popt, pcov = curve_fit(exponential_decay, t_data.reshape(-1), MAT[i].reshape(-1), p0=initial_guess)
    param.append(list(popt))

#%%
ind = 0
loss_exp = []
for ind in range(MAT.shape[0]):    
    plt.figure(dpi=300)
    
    #Datos
    plt.plot(t,MAT[ind], label='Valores Reales', color='blue', alpha=0.7)
    
    #Ajuste Exp
    y0 = param[ind][0]
    A1 = param[ind][1]
    t1 = param[ind][2]
    T_adjusted = exponential_decay(t, y0, A1, t1)
    plt.plot(t, T_adjusted, label=f'Ajuste Exponencial: $T(t)={y0:.2f}+{A1:.2f}exp(-t/{t1:.2f}$)', color='green', linewidth=2)
    ##Metrica:
    test_loss_exp = crit(torch.tensor(MAT[ind], dtype=torch.float32),torch.tensor(T_adjusted, dtype=torch.float32))
    loss_exp.append(test_loss_exp.item())
    #MLP
    plt.plot(t_data, MAT_pred[ind], label='Predicciones MLP', color='red', alpha=0.7) # Las predicciones de tu red neuronal

    plt.title('Comparación: MLP vs. Ajuste Exponencial vs. Valores Reales \n '+MAT_nom[ind])
    plt.xlabel('tiempo[h]')
    plt.ylabel('Temperatura [°C]')
    plt.legend()
    plt.grid(True)
    plt.savefig('comparacion'+MAT_nom[ind]+'.png')
    plt.show()


    '''
    Nuevo=np.hstack((np.array(param),
                     np.array(loss_exp).reshape(6,1),
                     np.array(loss_modelo).reshape(6,1)))
'''


#%% Isotermas
#Def Cuadricula
x_min, x_max = -0.054, 0.054
y_min, y_max = -0.054, 0.054
num_puntos_x = 50# Número de puntos en el eje X
num_puntos_y = 50# Número de puntos en el eje Y

# Crea grilla
X_grid, Y_grid = np.meshgrid(np.linspace(x_min, x_max, num_puntos_x),np.linspace(y_min, y_max, num_puntos_y))

for i in range(12):
    fixed_time=0.100+i*0.025
    fixed_T0 = 25.0
    fixed_q = 4.0
    
    points_x_flat = X_grid.flatten()
    points_y_flat = Y_grid.flatten()
    
    # Crear un array de NumPy para las entradas del modelo
    # Cada fila sera [timepo, x_coord, y_coord, fixed_T0, fixed_q]
    input_data_np = np.stack([
        np.full_like(points_x_flat, fixed_time), # Columna 't'
        points_x_flat,                           # Columna 'x'
        points_y_flat,                           # Columna 'y'
        np.full_like(points_x_flat, fixed_T0),   # Columna 'To'
        np.full_like(points_x_flat, fixed_q)     # Columna 'q'
    ], axis=1)
    
    # de array a un tensor de pytorch
    input_data_tensor = torch.tensor(input_data_np, dtype=torch.float32)
    
    model.eval()
    with torch.no_grad():
        predicted_temperatures_tensor = model(input_data_tensor) #valuo en le modelo
    
    #de tensor a array
    predicted_temperatures_np = predicted_temperatures_tensor.squeeze().numpy()
    
    #Reorganizacion de los datos para Graficar
    predicted_temperatures_grid = predicted_temperatures_np.reshape(num_puntos_y, num_puntos_x)
    
    #PLoTEO
    plt.figure(figsize=(10, 8))
    
    contourf_plot = plt.contourf(X_grid, Y_grid, predicted_temperatures_grid, levels=25, cmap='viridis') # 'levels'es  numero de isolíneas
    plt.colorbar(contourf_plot, label='Temperatura Predicha (T)')
    
    #lineas de contorno
    contour_lines = plt.contour(X_grid, Y_grid, predicted_temperatures_grid, levels=contourf_plot.levels, colors='k', linewidths=0.5)
    plt.clabel(contour_lines, inline=True, fontsize=8, fmt='%.1f') # Etiquetas para las líneas de contorno
    
    plt.title(f'Isotermas Predichas por MLP en t={fixed_time:.3f}, T0={fixed_T0:.2f}, q={fixed_q:.0f}')
    plt.xlabel('Coordenada X')
    plt.ylabel('Coordenada Y')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.savefig(f'Isoterma-{fixed_time:.3f}.png')
    plt.show()