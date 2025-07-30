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
path = r"D:\HAAGCD\Trabajo-Final\datos preprocesados"
path_arch = os.path.join(path,"DATOS2.csv")
df = pd.read_csv(path_arch, sep= ',', encoding='utf-8', dtype=np.float64).values
#%%
from sklearn.model_selection import train_test_split
X_train, X_testval, y_train, y_testval = train_test_split(df[:,0:5], df[:,5], train_size=0.5, shuffle=True)
X_test, X_val, y_test, y_val = train_test_split(X_testval, y_testval, train_size=0.5, shuffle=True)
#%%
#los vuelvo tensores de torch
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
        self.Lin1 = nn.Linear(n_in, n_hidd) 
        self.act =  torch.nn.Tanh()
        self.Lin2 = nn.Linear(n_hidd, n_hidd)
        self.act2 = torch.nn.Tanh()
        self.Lin3 = nn.Linear(n_hidd, n_out)
       
    def forward(self, x):
        salida = self.Lin3(self.act(self.Lin2(self.act2(self.Lin2(self.act2(self.Lin2(self.act2(self.Lin1(x)))))))))
              
        return salida
#%%
input_size = X_train.shape[1]# 5
output_size = y_train.shape[1] # 1

n_hidden_size = 64
model = Mired(input_size,n_hidden_size, output_size)
crit = nn.HuberLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
hist = []
#%%
num_epochs = 14000
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

print("--- Entrenamiento Finalizado ---")
plt.semilogy(hist)
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
    for i in range(20):
        print(f"Real: {y_test[i].item():.4f}, Predicho: {test_outputs[i].item():.4f}")
#%%
x_cheq_np = df[0:1740, 0:5] # Las características
y_cheq_np = df[0:1740, 5]

# Convertir a tensores de PyTorch (¡sin .requires_grad_()!)
x_cheq_tensor = torch.tensor(x_cheq_np, dtype=torch.float32)
y_cheq_tensor = torch.tensor(y_cheq_np, dtype=torch.float32).unsqueeze(1) # Añadir dimensión para que sea (N, 1)

# La primera columna de x_cheq_np es el "tiempo" (asumido para el eje X)
time_values = x_cheq_np[:, 0]

#Obtener las predicciones del modelo para estos datos
model.eval() # Poner el modelo en modo evaluación
with torch.no_grad(): # Desactivar el cálculo de gradientes
    predictions_tensor = model(x_cheq_tensor)
    cheq_loss = crit(predictions_tensor, y_cheq_tensor)

# Convertir las predicciones de tensor de PyTorch a array de NumPy para graficar
predictions_np = predictions_tensor.squeeze(1).numpy() # .squeeze(1) para quitar la dimensión extra si es (N, 1) y convertir a (N,)

# 3. Crear la gráfica
plt.figure(figsize=(12, 6)) # Tamaño de la figura para mejor visualización

# Graficar los valores reales
plt.plot(time_values, y_cheq_np, label='Valores Reales', color='blue', alpha=0.7)

# Graficar las predicciones
plt.plot(time_values, predictions_np, label='Predicciones del Modelo', color='red', linestyle='--', alpha=0.7)

plt.title('Predicciones del Modelo vs. Valores Reales')
plt.xlabel('Tiempo (primera columna de entrada)')
plt.ylabel('Temperatura (salida T)')
plt.legend() # Muestra la leyenda de las etiquetas
plt.grid(True) # Añade una cuadrícula al gráfico
plt.show() # Muestra la ventana de la gráfica
print(f'Pérdida en el conjunto de prueba: {cheq_loss.item():.4f}')

#%% Ajuste exponencial

from scipy.optimize import curve_fit
t_data = x_cheq_np[:, 0]
T_data = y_cheq_np

#Def fun exponencial para el ajuste
def exponential_decay(t, y0, A1, t1):
    """
    Función de decaimiento exponencial: T(t) = y0 + A1 * exp(-t/t1)
    """
    return y0 + A1 * np.exp(-t / t1)


initial_guess = [T_data.min(), T_data.max() - T_data.min(), np.mean(t_data) / 2]

# curve_fit devuelve los parámetros óptimos y la matriz de covarianza
popt, pcov = curve_fit(exponential_decay, t_data, T_data, p0=initial_guess)
y0_opt, A1_opt, t1_opt = popt
'''
    print(f"Parámetros del ajuste exponencial: ")
    print(f"  y0 = {y0_opt:.4f}")
    print(f"  A1 = {A1_opt:.4f}")
    print(f"  t1 = {t1_opt:.4f}")
'''
#Calcular los valores predichos por el modelo ajustado
T_adjusted = exponential_decay(t_data, y0_opt, A1_opt, t1_opt)

T_adjusted_tensor = torch.tensor(T_adjusted, dtype=torch.float32).unsqueeze(1)
T_data_tensor = torch.tensor(T_data, dtype=torch.float32).unsqueeze(1) 
#Calcular el "loss" (error) entre el ajuste y los datos reales
# Puedes usar el Error Cuadrático Medio (MSE) o el Error Absoluto Medio (MAE)
loss_adjustment_mse = np.mean((T_data - T_adjusted)**2)
loss_adjustment_mae = np.mean(np.abs(T_data - T_adjusted))
loss_ajuste = crit(T_data_tensor,T_adjusted_tensor)

print(f"\nPérdida del Ajuste Exponencial (MSE): {loss_adjustment_mse:.4f}")
print(f"Pérdida del Ajuste Exponencial (MAE): {loss_adjustment_mae:.4f}")
print(f"Pérdida del Ajuste Exponencial (crit): {loss_ajuste:.4f}")
print(f'Pérdida en el conjunto de prueba con modelo: {cheq_loss.item():.4f}')



#Graficar el ajuste junto con los datos reales y las predicciones de la NN
plt.figure(figsize=(12, 6))
plt.plot(t_data, T_data, label='Valores Reales (y_cheq)', color='blue', alpha=0.7)
plt.plot(t_data, predictions_np, label='Predicciones MLP', color='red', linestyle='--', alpha=0.7) # Las predicciones de tu red neuronal
plt.plot(t_data, T_adjusted, label=f'Ajuste Exponencial: T(t)={y0_opt:.2f}+{A1_opt:.2f}e^(-t/{t1_opt:.2f})', color='green', linestyle='-', linewidth=2)

plt.title('Comparación: MLP vs. Ajuste Exponencial vs. Valores Reales')
plt.xlabel('Tiempo (t)')
plt.ylabel('Temperatura (T)')
plt.legend()
plt.grid(True)
plt.show()
    
#%%
import numpy as np
import torch
import matplotlib.pyplot as plt
# Asegúrate de que tu modelo 'Mired' y el modelo entrenado 'model' ya están definidos y entrenados

print("\n--- Generando y Visualizando Isotermas Predichas ---")

# --- 1. Definir la Cuadrícula (Red de Puntos) ---
# Establece los rangos de las coordenadas x e y
# Ajusta estos valores según el dominio espacial que deseas visualizar
x_min, x_max = -0.054, 0.054 # Ejemplo: Rango de X
y_min, y_max = -0.054, 0.054 # Ejemplo: Rango de Y
num_points_x = 50      # Número de puntos en el eje X
num_points_y = 50      # Número de puntos en el eje Y

# Crea las mallas de coordenadas X e Y
X_grid, Y_grid = np.meshgrid(np.linspace(x_min, x_max, num_points_x),
                             np.linspace(y_min, y_max, num_points_y))

# Define los valores fijos para t, To, y q para esta visualización
# Deben ser escalares, por ejemplo, el tiempo 't' en un instante específico
fixed_time = 0.150  # Un instante de tiempo específico dentro de tu rango de datos
fixed_T0 = 25.0    # Valor inicial de temperatura
fixed_q = 4.0      # Valor de alguna fuente o sumidero (si aplica a tus datos)

# --- 2. Crear Datos de Entrada para el Modelo ---
# Aplanar las mallas para crear un array de puntos (x, y)
points_x_flat = X_grid.flatten()
points_y_flat = Y_grid.flatten()

# Crear un array de NumPy para las entradas del modelo
# Cada fila será [fixed_time, x_coord, y_coord, fixed_T0, fixed_q]
# Asegúrate de que el orden de las columnas coincida con el orden de entrada de tu red (t, x, y, To, q)
input_data_np = np.stack([
    np.full_like(points_x_flat, fixed_time), # Columna 't'
    points_x_flat,                           # Columna 'x'
    points_y_flat,                           # Columna 'y'
    np.full_like(points_x_flat, fixed_T0),   # Columna 'To'
    np.full_like(points_x_flat, fixed_q)     # Columna 'q'
], axis=1)

# Convertir el array de NumPy a un tensor de PyTorch
input_data_tensor = torch.tensor(input_data_np, dtype=torch.float32)

# --- 3. Realizar Predicciones ---
# Poner el modelo en modo evaluación y desactivar el cálculo de gradientes
model.eval()
with torch.no_grad():
    predicted_temperatures_tensor = model(input_data_tensor)

# Convertir las predicciones de tensor de PyTorch a array de NumPy
# .squeeze() para eliminar cualquier dimensión extra y asegurar que sea 1D
predicted_temperatures_np = predicted_temperatures_tensor.squeeze().numpy()

# --- 4. Reorganizar los Datos para Graficar (Volver a la Forma de Malla) ---
# Las predicciones ahora son un vector plano. Necesitamos darle la forma de la malla (num_points_x, num_points_y)
# np.reshape() es perfecto para esto.
predicted_temperatures_grid = predicted_temperatures_np.reshape(num_points_y, num_points_x)

# --- 5. Visualizar las Isotermas ---
plt.figure(figsize=(10, 8)) # Tamaño de la figura

# plt.contourf() para crear un mapa de colores de las isotermas rellenas
# plt.contour() para dibujar las líneas de contorno (opcional, pero útil para ver los niveles exactos)
contourf_plot = plt.contourf(X_grid, Y_grid, predicted_temperatures_grid, levels=25, cmap='viridis') # 'levels' controla el número de isolíneas/colores
plt.colorbar(contourf_plot, label='Temperatura Predicha (T)') # Añade la barra de color

# Puedes añadir líneas de contorno si lo deseas
# contour_lines = plt.contour(X_grid, Y_grid, predicted_temperatures_grid, levels=contourf_plot.levels, colors='k', linewidths=0.5)
# plt.clabel(contour_lines, inline=True, fontsize=8, fmt='%.1f') # Etiquetas para las líneas de contorno

plt.title(f'Isotermas Predichas por MLP en t={fixed_time}, T0={fixed_T0}, q={fixed_q}')
plt.xlabel('Coordenada X')
plt.ylabel('Coordenada Y')
plt.grid(True, linestyle=':', alpha=0.6)
plt.show()

print("--- Generación y Visualización de Isotermas Finalizada ---")



"""
#%%GUARDADO DE RED
torch.save(model.state_dict(),"mired2_15milepocas.pth")
#%% CARGA DE RED       
SDPATH = torch.load("mired2_15milepocas.pth")
SDPATH.keys()
model.load_state_dict(SDPATH)
"""

#%%
# --- 1. Definir la Cuadrícula (Red de Puntos) ---
# Establece los rangos de las coordenadas x e y
# Ajusta estos valores según el dominio espacial que deseas visualizar
x_min, x_max = -0.054, 0.054 # Ejemplo: Rango de X
y_min, y_max = -0.054, 0.054 # Ejemplo: Rango de Y
num_points_x = 50      # Número de puntos en el eje X
num_points_y = 50      # Número de puntos en el eje Y

# Crea las mallas de coordenadas X e Y
X_grid, Y_grid = np.meshgrid(np.linspace(x_min, x_max, num_points_x),
                             np.linspace(y_min, y_max, num_points_y))

# Define los valores fijos para t, To, y q para esta visualización
# Deben ser escalares, por ejemplo, el tiempo 't' en un instante específico
#fixed_time = 0.150  # Un instante de tiempo específico dentro de tu rango de datos
for i in range(12):
    fixed_time=0.100+i*0.025
    fixed_T0 = 25.0    # Valor inicial de temperatura
    fixed_q = 4.0      # Valor de alguna fuente o sumidero (si aplica a tus datos)
    
    # --- 2. Crear Datos de Entrada para el Modelo ---
    # Aplanar las mallas para crear un array de puntos (x, y)
    points_x_flat = X_grid.flatten()
    points_y_flat = Y_grid.flatten()
    
    # Crear un array de NumPy para las entradas del modelo
    # Cada fila será [fixed_time, x_coord, y_coord, fixed_T0, fixed_q]
    # Asegúrate de que el orden de las columnas coincida con el orden de entrada de tu red (t, x, y, To, q)
    input_data_np = np.stack([
        np.full_like(points_x_flat, fixed_time), # Columna 't'
        points_x_flat,                           # Columna 'x'
        points_y_flat,                           # Columna 'y'
        np.full_like(points_x_flat, fixed_T0),   # Columna 'To'
        np.full_like(points_x_flat, fixed_q)     # Columna 'q'
    ], axis=1)
    
    # Convertir el array de NumPy a un tensor de PyTorch
    input_data_tensor = torch.tensor(input_data_np, dtype=torch.float32)
    
    # --- 3. Realizar Predicciones ---
    # Poner el modelo en modo evaluación y desactivar el cálculo de gradientes
    model.eval()
    with torch.no_grad():
        predicted_temperatures_tensor = model(input_data_tensor)
    
    # Convertir las predicciones de tensor de PyTorch a array de NumPy
    # .squeeze() para eliminar cualquier dimensión extra y asegurar que sea 1D
    predicted_temperatures_np = predicted_temperatures_tensor.squeeze().numpy()
    
    # --- 4. Reorganizar los Datos para Graficar (Volver a la Forma de Malla) ---
    # Las predicciones ahora son un vector plano. Necesitamos darle la forma de la malla (num_points_x, num_points_y)
    # np.reshape() es perfecto para esto.
    predicted_temperatures_grid = predicted_temperatures_np.reshape(num_points_y, num_points_x)
    
    # --- 5. Visualizar las Isotermas ---
    plt.figure(figsize=(10, 8)) # Tamaño de la figura
    
    # plt.contourf() para crear un mapa de colores de las isotermas rellenas
    # plt.contour() para dibujar las líneas de contorno (opcional, pero útil para ver los niveles exactos)
    contourf_plot = plt.contourf(X_grid, Y_grid, predicted_temperatures_grid, levels=25, cmap='viridis') # 'levels' controla el número de isolíneas/colores
    plt.colorbar(contourf_plot, label='Temperatura Predicha (T)') # Añade la barra de color
    
    # Puedes añadir líneas de contorno si lo deseas
    # contour_lines = plt.contour(X_grid, Y_grid, predicted_temperatures_grid, levels=contourf_plot.levels, colors='k', linewidths=0.5)
    # plt.clabel(contour_lines, inline=True, fontsize=8, fmt='%.1f') # Etiquetas para las líneas de contorno
    
    plt.title(f'Isotermas Predichas por MLP en t={fixed_time}, T0={fixed_T0}, q={fixed_q}')
    plt.xlabel('Coordenada X')
    plt.ylabel('Coordenada Y')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.show()