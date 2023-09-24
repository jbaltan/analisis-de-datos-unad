import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Cargar los datos desde el archivo CSV
ruta_csv = "D:/OneDrive - Universidad Nacional Abierta y a Distancia/universidad/Sexto semestre/ANÁLISIS DE DATOS/actividad 2/dataset/regresion-lineal.csv"
data = pd.read_csv(ruta_csv)

# Extraer las columnas 'metro' (característica) y 'precio' (etiqueta)
metros = data['metro'].values.reshape(-1, 1)
precios = data['precio'].values

# Crear un modelo de regresión lineal
modelo = LinearRegression()

# Entrenar el modelo con los datos
modelo.fit(metros, precios)

# Realizar predicciones
predicciones = modelo.predict(metros)

# Visualizar los resultados
plt.scatter(metros, precios, label='Datos originales')
plt.plot(metros, predicciones, color='red', label='Línea de regresión')
plt.xlabel('Metro')
plt.ylabel('Precio')
plt.legend()
plt.show()

# Imprimir los coeficientes del modelo
print("Intersección (b):", modelo.intercept_)
print("Pendiente (m):", modelo.coef_)
