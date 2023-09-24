# Importar las bibliotecas necesarias
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Cargar el archivo CSV
csv_file = r'D:/OneDrive - Universidad Nacional Abierta y a Distancia/universidad/Sexto semestre/ANÁLISIS DE DATOS/actividad 2/solucion/regresion logistica/regresion-logistica.csv'
df = pd.read_csv(csv_file)

# Imputar los valores faltantes con la media
imputer = SimpleImputer(strategy='mean')
df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

# Dividir los datos en características (X) y la variable objetivo (y)
X = df.drop('TenYearCHD', axis=1)
y = df['TenYearCHD']

# Escalar los datos para que tengan media cero y desviación estándar uno
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Dividir el conjunto de datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Crear un modelo de regresión logística
model = LogisticRegression(max_iter=1000)

# Entrenar el modelo
model.fit(X_train, y_train)

# Hacer predicciones en el conjunto de prueba
y_pred = model.predict(X_test)

# Evaluar el rendimiento del modelo
print("Matriz de Confusión:")
print(confusion_matrix(y_test, y_pred))
print("\nInforme de Clasificación:")
print(classification_report(y_test, y_pred))

# Visualizar los coeficientes de la regresión logística
coefficients = model.coef_[0]
feature_names = X.columns
plt.figure(figsize=(10, 6))
plt.barh(feature_names, coefficients)
plt.xlabel('Coeficiente')
plt.ylabel('Característica')
plt.title('Coeficientes de la Regresión Logística')
plt.show()

