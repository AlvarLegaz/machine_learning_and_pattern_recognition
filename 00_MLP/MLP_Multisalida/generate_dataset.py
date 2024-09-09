import pandas as pd
import numpy as np

# Configuración de parámetros
np.random.seed(42)
num_samples = 150

# Generar datos aleatorios
peso = np.random.uniform(45, 100, num_samples)  # Peso entre 45 y 100 kg
altura = np.random.uniform(150, 200, num_samples)  # Altura entre 150 y 200 cm
genero = np.random.choice(['M', 'F'], num_samples)  # Género M o F
nivel_grasa = np.random.uniform(10, 35, num_samples)  # Nivel de grasa entre 10% y 35%

# Clasificación basada en el nivel de grasa, eliminando "En Forma"
clasificacion = np.where(nivel_grasa < 18, 'Delgado', 
                np.where(nivel_grasa < 25, 'Normal', 'Sobrepeso'))

# Crear DataFrame
data = pd.DataFrame({
    'Peso (kg)': peso,
    'Altura (cm)': altura,
    'Género': genero,
    'Nivel de Grasa (%)': nivel_grasa,
    'Clasificación': clasificacion
})

# Filtrar para eliminar la categoría "En Forma"
data = data[data['Clasificación'] != 'En Forma']

# Guardar a un archivo Excel
data.to_excel('dataset.xlsx', index=False)

print("Conjunto de datos generado y guardado en 'conjunto_datos_150_muestras_sin_en_forma.xlsx'")
