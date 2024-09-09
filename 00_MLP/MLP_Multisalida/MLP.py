import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Desactiva los mensajes de información y advertencia
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt

# Cargar datos
data = pd.read_excel('dataset.xlsx')

# Preprocesamiento
label_encoder = LabelEncoder() 
data['Género'] = label_encoder.fit_transform(data['Género']) # para convertir las etiquetas categóricas en números.
data['Clasificación'] = label_encoder.fit_transform(data['Clasificación']) # para convertir las etiquetas categóricas en números.

X = data[['Peso (kg)', 'Altura (cm)', 'Género', 'Nivel de Grasa (%)']]
y = data['Clasificación']

#Dividimos conjunto entrenamiento (80%) y test (20%)
# - train_test_split: para dividir los datos en conjuntos de entrenamiento y prueba.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) 

# Crear el modelo
# - 2 capas ocultas densas (fully connected layers)
# 	- 1 capa oculta Esta capa puede aprender características básicas de los datos, como combinaciones lineales de las entradas (peso, 	altura, género, nivel de grasa).
#	- 2 Segunda capa oculta (8 neuronas): Esta capa puede tomar las características aprendidas por la primera capa y combinarlas para aprender patrones más complejos y no lineales. Por ejemplo, puede aprender interacciones entre el peso y el nivel de grasa que son más complejas que una simple combinación lineal.
# - capa salida con 4 neuronas, 1 por clase.
model = Sequential()
model.add(Dense(16, input_dim=4, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(4, activation='softmax'))

# Compilar el modelo
# - Loss Function: Usamos sparse_categorical_crossentropy porque estamos trabajando con etiquetas categóricas.
# - Optimizer: Usamos adam, que es un optimizador eficiente.
# - Metrics: Evaluamos el modelo usando la precisión (accuracy).
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Entrenar el modelo y guardar el historial
# - epoch: define el número de veces que el algoritmo de entrenamiento pasará por todo el conjunto de datos de entrenamiento
# - barch_siz: define el número de muestras que se utilizarán en cada actualización de los pesos del modelo. Aquí, el modelo actualizará sus pesos después de cada 10 muestras. Si no se pone nada esta por defecto a 3.
history = model.fit(X_train, y_train, epochs=150, batch_size=10, validation_data=(X_test, y_test))

# Evaluar el modelo
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Precisión: {accuracy * 100:.2f}%')

# Graficar la evolución del error y la precisión
plt.figure(figsize=(12, 5))

# Gráfica de la pérdida
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Pérdida de entrenamiento')
plt.plot(history.history['val_loss'], label='Pérdida de validación')
plt.xlabel('Épocas')
plt.ylabel('Pérdida')
plt.legend()
plt.title('Evolución de la Pérdida')

# Gráfica de la precisión
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Precisión de entrenamiento')
plt.plot(history.history['val_accuracy'], label='Precisión de validación')
plt.xlabel('Épocas')
plt.ylabel('Precisión')
plt.legend()
plt.title('Evolución de la Precisión')

plt.show()
