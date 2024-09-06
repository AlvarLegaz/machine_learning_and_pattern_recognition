import numpy as np
import pandas as pd

# Datos de entrada
X = np.array([
    [0.1, 0.2],
    [0.4, 0.6],
    [0.7, 0.3],
    [0.9, 0.8],
    [0.2, 0.9],
    [0.5, 0.5],
    [0.8, 0.1],
    [0.3, 0.7],
    [0.15, 0.85],
    [0.65, 0.35],
    [0.25, 0.75],
    [0.55, 0.45],
    [0.05, 0.95],
    [0.45, 0.55],
    [0.85, 0.15],
    [0.35, 0.65],
    [0.75, 0.25],
    [0.95, 0.05],
    [0.6, 0.4],
    [0.1, 0.9]
])

# Datos de salida
y = np.array([
    0, 0, 1, 0, 1, 0, 1, 1, 1, 1,
    1, 0, 1, 0, 1, 1, 1, 1, 0, 1
])

# Crear un DataFrame
df = pd.DataFrame(X, columns=['Entrada 1', 'Entrada 2'])
df['Salida'] = y

# Guardar en un archivo CSV
df.to_csv('xor_data.csv', index=False)
