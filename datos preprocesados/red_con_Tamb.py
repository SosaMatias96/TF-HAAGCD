#%% imports
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
#%% Carga de archivos
path = r"D:\HAAGCD\Trabajo-Final\datos preprocesados"
path_arch = os.path.join(path,"DATOS_con_Tamb.csv")
df = pd.read_csv(path_arch, sep= ';', encoding='utf-8', dtype=np.float64).values
#%%
from sklearn.model_selection import train_test_split
X_train, X_testval, y_train, y_testval = train_test_split(df[:,0:6], df[:,6], train_size=0.9, shuffle=True)
X_test, X_val, y_test, y_val = train_test_split(X_testval, y_testval, train_size=0.5, shuffle=True)
#%%
#los vuelvo tensores de torch
X_train = torch.FloatTensor(X_train).requires_grad_().view(-1,6)
X_test = torch.FloatTensor(X_test).requires_grad_().view(-1,6)
X_val = torch.FloatTensor(X_val).requires_grad_().view(-1,6)

y_train = torch.FloatTensor(y_train).requires_grad_().view(-1,1)
y_test = torch.FloatTensor(y_test).requires_grad_().view(-1,1)
y_val = torch.FloatTensor(y_val).requires_grad_().view(-1,1)
#X_train.shape -->torch.Size([8535, 5])
#%%RED NEURONAL
#entrada: t,x,y,To,q  salida: T

class Mired(nn.Module):
   
    def __init__(self, n_in, n_hidd, n_out):
      
        super().__init__()
        self.Lin1 = nn.Linear(n_in, n_hidd) 
        self.act =  torch.nn.Tanh()
        self.Lin2 = nn.Linear(n_hidd, n_hidd)
        self.act2 = torch.nn.Tanh()
        self.Lin3 = nn.Linear(n_hidd, n_out)
       
    def forward(self, x):
        salida = self.Lin3(self.act(self.Lin2(self.act2(self.Lin1(x)))))  
              
        return salida
#%%
input_size = X_train.shape[1]# 5
output_size = y_train.shape[1] # 1

n_hidden_size = 16

model = Mired(input_size,n_hidden_size, output_size)
crit = nn.HuberLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
hist = []
losses = []
#%%
num_epochs = 5000
for epoch in range(num_epochs):
    # Pone el modelo en modo entrenamiento
    model.train()

    # Forward pass: calcular las predicciones
    outputs = model(X_train)

    # Calcular la pérdida
    loss = crit(outputs, y_train)

    # Backward pass: calcular gradientes
    optimizer.zero_grad() # Limpiar los gradientes anteriores
    loss.backward()       # Calcular los gradientes
    optimizer.step()      # Actualizar los pesos del modelo

    #Validacion
    model.eval() # Poner el modelo en modo evaluación (deshabilita Dropout, etc.)
    with torch.no_grad(): # Desactivar el cálculo de gradientes para la validación (ahorra memoria y tiempo)
        val_outputs = model(X_val)
        val_loss = crit(val_outputs, y_val)
        hist.append(val_loss.item())

    if (epoch + 1) % 10 == 0: # Imprimir loss cada 10 épocas
        print(f'Epoch [{epoch+1}/{num_epochs}], Pérdida de Entrenamiento: {loss.item():.4f}, Pérdida de Validación: {val_loss.item():.4f}')
        losses.append([epoch,loss.item(),val_loss.item()])
print("--- Entrenamiento Finalizado ---")
plt.semilogy(hist)
plt.show()
losses=np.array(losses)
plt.plot(losses[:,0],losses[:,1],label="entr")
plt.plot(losses[:,0],losses[:,2],label="val")
plt.legend()
plt.show()
#%%
print("\n--- Evaluación en el Conjunto de Prueba ---")
model.eval() # Poner el modelo en modo evaluación
with torch.no_grad(): # Desactivar el cálculo de gradientes
    test_outputs = model(X_test)
    test_loss = crit(test_outputs, y_test)

    print(f'Pérdida en el conjunto de prueba: {test_loss.item():.4f}')

    # Si quisieras ver algunas predicciones vs. valores reales (solo las primeras 5):
    print("\nEjemplo de Predicciones vs. Valores Reales (primeras 5):")
    for i in range(10):
        print(f"Real: {y_test[i].item():.4f}, Predicho: {test_outputs[i].item():.4f}")

#%%GUARDADO DE RED
torch.save(model.state_dict(),"mired2_15milepocas.pth")
#%% CARGA DE RED       
SDPATH = torch.load("mired2_15milepocas.pth")
SDPATH.keys()
model.load_state_dict(SDPATH)