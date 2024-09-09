import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Cargar datos
data = pd.read_excel('ruta_a_tu_archivo.xlsx')

# Preprocesamiento
label_encoder = LabelEncoder()
data['Género'] = label_encoder.fit_transform(data['Género'])
data['Clasificación'] = label_encoder.fit_transform(data['Clasificación'])

X = data[['Peso (kg)', 'Altura (cm)', 'Género', 'Nivel de Grasa (%)']]
y = data['Clasificación']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear el modelo
model = Sequential()
model.add(Dense(16, input_dim=4, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(4, activation='softmax'))

# Compilar el modelo
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Entrenar el modelo
model.fit(X_train, y_train, epochs=50, batch_size=10, validation_data=(X_test, y_test))

# Evaluar el modelo
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Precisión: {accuracy * 100:.2f}%')
