"""
RED NEURONAL--> Entrenamiento
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
#%% Cargo/abro los datos
path = r"D:\HAAGCD\Trabajo-Final\datos preprocesados" #cargo el path de los datos
path_arch = os.path.join(path,"DATOS3.csv")
df = pd.read_csv(path_arch, sep= ',', encoding='utf-8', dtype=np.float64).values #abro el archivo con pandas
#OJO! el de Tamb tiene separados ';'
#%% Separo el set de datos en 70%train, 15% para test, 15% ara valid
from sklearn.model_selection import train_test_split
#separo el set de train y mezclo
X_train, X_testval, y_train, y_testval = train_test_split(df[:,0:5], df[:,5], train_size=0.5, shuffle=True)
#Re-separo en test y val
X_test, X_val, y_test, y_val = train_test_split(X_testval, y_testval, train_size=0.5, shuffle=True)
#%% Los vuelvo tensores de torch
X_train = torch.FloatTensor(X_train).requires_grad_().view(-1,5)
X_test = torch.FloatTensor(X_test).requires_grad_().view(-1,5)
X_val = torch.FloatTensor(X_val).requires_grad_().view(-1,5)

y_train = torch.FloatTensor(y_train).requires_grad_().view(-1,1)
y_test = torch.FloatTensor(y_test).requires_grad_().view(-1,1)
y_val = torch.FloatTensor(y_val).requires_grad_().view(-1,1)
#X_train.shape -->torch.Size([8535, 5])

#%%RED NEURONAL
#entrada: t,x,y,To,q  salida: T

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
#%% Creacion de la red
#def de tamaños
input_size = X_train.shape[1]# 5
output_size = y_train.shape[1] # 1

n_hidden_size = 64
#instancio la red
model = Mired(input_size,n_hidden_size, output_size)
#declaro la metrica y el optimizador
crit = nn.HuberLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
#%% Entrenamiento
hist = []
val =[]
num_epochs = 14000 #con 5000 esta hasta sobrada
for epoch in range(num_epochs):
    # Modo entrenamiento
    model.train()
    # Forward pass
    outputs = model(X_train)
    # Calculo la loss
    loss = crit(outputs, y_train)
    # Backward pass(gradientes)
    optimizer.zero_grad() # Limpiar los gradientes anteriores
    loss.backward()       # Calcular los gradientes
    optimizer.step()      # Actualizar los pesos del modelo

    #Validacion
    model.eval() # Poner el modelo en modo evaluacion
    with torch.no_grad(): # Desactivar el cálculo de gradientes para la validacion
        val_outputs = model(X_val)
        val_loss = crit(val_outputs, y_val)
        hist.append(val_loss.item())
        val.append([epoch,loss.item(),val_loss.item()])

    if (epoch) % 10 == 0: # Imprimir loss cada 10 epocas
        print(f'Epoch [{epoch}/{num_epochs}], Pérdida de Entrenamiento: {loss.item():.4f}, Pérdida de Validación: {val_loss.item():.4f}')

print("Ya ta")
plt.semilogy(hist)
plt.title('Evolucion de Loss')
plt.xlabel('Epoca')
plt.ylabel('Loss')
#plt.legend()
plt.grid(True) # Añade una cuadrícula al gráfico
plt.show()

val=np.array(val)
plt.semilogy(val[:,0],val[:,1],'x',label = 'train')
plt.semilogy(val[:,0],val[:,2],label = 'val')
plt.legend()
plt.xlabel('Epoca')
plt.ylabel('Loss')
plt.show()
#%% Prueba de modelo con conjunto de Testeo
model.eval() # Poner el modelo en modo evaluacion
with torch.no_grad(): # Desactiva el calculo de gradientes
    test_outputs = model(X_test) #Valuo modelo en el test
    test_loss = crit(test_outputs, y_test) #calculo metrica

    print(f'Pérdida en el conjunto de prueba: {test_loss.item():.4f}')

    # Si quisieras ver algunas predicciones vs. valores reales (solo las primeras 5):
    print("Valores Reales vs. Valores Predichos:")
    for i in range(20):
        print(f"Real: {y_test[i].item():.4f}, Predicho: {test_outputs[i].item():.4f}")
#%% Grafica del TEST
test_outputs_np=test_outputs.detach().numpy()
y_test_np=y_test.detach().numpy()
x=np.linspace(min(y_test_np),max(y_test_np))
plt.figure(figsize=(6, 3),dpi=300)
plt.scatter(y_test_np,test_outputs_np,s=5,alpha=0.25 )
plt.plot(x,x,color="black")
plt.xlabel('Dato')
plt.ylabel('Prediccion')
plt.show()
#%%
plt.figure(figsize=(6, 3))
errores = y_test_np - test_outputs_np
media_errores = np.mean(errores)
std_errores = np.std(errores)
plt.hist(errores, bins=50, edgecolor='black', alpha=0.7)
plt.axvline(0, color='red', linestyle='--', linewidth=2)
plt.title(f'Histograma de Errores de Predicción en el Conjunto de Test \n media:{media_errores:.3f}°C $\sigma = $ {std_errores:.3f}°C', fontsize=12)
plt.xlabel('Error de Predicción (°C)', fontsize=12)
plt.ylabel('Frecuencia', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.xlim(-0.4,0.4)
plt.show()

#%% Guardo la red
torch.save(model.state_dict(),"mired-4capas-64hidden_14mil-epocas.pth")