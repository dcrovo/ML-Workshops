import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
import os 

abs_path = os.getcwd()
modelo_f = os.path.join(abs_path,'classifier.h5')
scaler_f = os.path.join(abs_path,'mi_scaler.pkl')
data_f = os.path.join(abs_path,'quiz_v2.csv')


modelo = Sequential([
    Dense(128, activation='relu', input_shape=(33,)),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(16, activation='relu'),
    Dense(8, activation='relu'),    
    Dense(4, activation='relu'),
    Dense(1, activation='sigmoid')
])
modelo.load_weights(modelo_f)
scaler_cargado = joblib.load(scaler_f)

#  Predecir con el conjunto quiz_v2.csv
# Cargar los datos
quiz_data = pd.read_csv(data_f)
quiz_data_scaled = scaler_cargado.transform(quiz_data)
predicciones_quiz = modelo.predict(quiz_data_scaled)
etiquetas_predichas = (predicciones_quiz > 0.5).astype(int).flatten()
# Guardar las predicciones
np.savetxt('answers.txt', etiquetas_predichas, fmt='%d')