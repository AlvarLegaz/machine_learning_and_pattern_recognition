# Este script entrena una red neuronal para resolver el problema XOR utilizando TensorFlow y Keras.
# El modelo tiene una capa oculta con 16 neuronas y una capa de salida con 1 neurona.
# La capa oculta utiliza la función de activación 'relu' y la capa de salida utiliza 'sigmoid'.
# El modelo se entrena durante 1000 épocas y se evalúa su precisión en los datos de entrada.

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Desactiva los mensajes de información y advertencia
import tensorflow as tf
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

# Leer el archivo CSV
df = pd.read_csv('xor_data.csv')

# Mostrar el contenido del archivo
print(df)

# Definir X e y
X = df[['Entrada 1', 'Entrada 2']].values
y = df['Salida'].values

# Parámetros del modelo
Num_neuronas = 16
epocas = 1000  # Iteraciones del entrenamiento

# Definir el modelo
model = Sequential()
model.add(Dense(Num_neuronas, input_dim=2, activation='relu'))  # Capa oculta con 100 neuronas
model.add(Dense(1, activation='sigmoid'))  # Capa de salida

# Compilar el modelo
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Entrenar el modelo
model.fit(X, y, epochs=epocas, verbose=0)

# Evaluar el modelo
loss, accuracy = model.evaluate(X, y)
print(f'Precisión: {accuracy*100:.2f}%')

# Hacer predicciones
predictions = model.predict(X)
print(np.round(predictions))
