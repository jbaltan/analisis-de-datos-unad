import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report

# Cargar los datos desde el archivo "wine.data"
column_names = ["Class", "Alcohol", "Malic acid", "Ash", "Alcalinity of ash", "Magnesium", "Total phenols", "Flavanoids", "Nonflavanoid phenols", "Proanthocyanins", "Color intensity", "Hue", "OD280/OD315 of diluted wines", "Proline"]
data = pd.read_csv("D:/OneDrive - Universidad Nacional Abierta y a Distancia/universidad/Sexto semestre/ANÁLISIS DE DATOS/actividad 2/solucion/arboles de decision/wine.data", names=column_names)

# Dividir los datos en características (X) y etiquetas (y)
X = data.drop("Class", axis=1)
y = data["Class"]

# Validación cruzada estratificada con 5 particiones
stratified_kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Inicializar una lista para almacenar las puntuaciones de precisión en cada partición
accuracy_scores = []

# Realizar validación cruzada
for train_index, test_index in stratified_kfold.split(X, y):
    X_train, X_val = X.iloc[train_index], X.iloc[test_index]
    y_train, y_val = y.iloc[train_index], y.iloc[test_index]

    # Crear un modelo de árbol de decisión
    model = DecisionTreeClassifier(random_state=42)

    # Entrenar el modelo con los datos de entrenamiento
    model.fit(X_train, y_train)

    # Realizar predicciones en el conjunto de validación
    y_pred_val = model.predict(X_val)

    # Calcular la precisión en esta partición y agregarla a la lista
    accuracy = accuracy_score(y_val, y_pred_val)
    accuracy_scores.append(accuracy)

# Calcular la precisión promedio de todas las particiones
average_accuracy = sum(accuracy_scores) / 5
print("Precisión promedio con validación cruzada:", average_accuracy)

# Definir una cuadrícula de hiperparámetros para ajustar
param_grid = {
    'max_depth': [None, 10, 20, 30],
    'criterion': ['gini', 'entropy']
}

# Inicializar GridSearchCV con validación cruzada
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=stratified_kfold)

# Realizar la búsqueda de cuadrícula en los datos de entrenamiento
grid_search.fit(X_train, y_train)

# Obtener los mejores hiperparámetros encontrados
best_params = grid_search.best_params_
print("Mejores hiperparámetros:", best_params)

# Entrenar el modelo con los mejores hiperparámetros en todos los datos de entrenamiento
best_model = grid_search.best_estimator_
best_model.fit(X_train, y_train)

# Realizar predicciones en el conjunto de prueba con el mejor modelo
y_pred_test = best_model.predict(X_test)

# Calcular la precisión del modelo con los mejores hiperparámetros
accuracy_best_model = accuracy_score(y_test, y_pred_test)
print("Precisión del modelo con mejores hiperparámetros:", accuracy_best_model)

# Visualizar el árbol de decisión resultante
plt.figure(figsize=(15, 10))
plot_tree(best_model, feature_names=column_names[1:], class_names=[str(cls) for cls in best_model.classes_], filled=True)
plt.show()

# Reporte de clasificación con métricas adicionales
classification_rep = classification_report(y_test, y_pred_test, target_names=[str(cls) for cls in best_model.classes_])
print("Reporte de clasificación:\n", classification_rep)
