import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

# Datos de entrada
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
# Salidas esperadas
y = np.array([[0], [1], [1], [0]])

# Definir el modelo
model = Sequential()
model.add(Dense(2, input_dim=2, activation='relu'))  # Capa oculta con 2 neuronas
model.add(Dense(1, activation='sigmoid'))  # Capa de salida

# Compilar el modelo
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Entrenar el modelo
model.fit(X, y, epochs=1000, verbose=0)

# Evaluar el modelo
loss, accuracy = model.evaluate(X, y)
print(f'Precisión: {accuracy*100}%')

# Hacer predicciones
predictions = model.predict(X)
print(np.round(predictions))
