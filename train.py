# Este archivo (train.py) contiene el código para entrenar un modelo de clasificación de flores iris

# Importamos las bibliotecas necesarias:
import pandas as pd  # Para manipular y analizar datos en formato tabular
import numpy as np  # Para operaciones numéricas y arrays
from sklearn.model_selection import train_test_split, GridSearchCV  # Para dividir datos y buscar mejores parámetros
from sklearn.ensemble import RandomForestClassifier  # El algoritmo de clasificación que usaremos
from sklearn.preprocessing import StandardScaler  # Para escalar/normalizar los datos
from sklearn.pipeline import Pipeline  # Para crear un flujo de procesamiento de datos
import joblib  # Para guardar y cargar modelos entrenados
import os  # Para operaciones con archivos y directorios
from tqdm import tqdm  # Para mostrar barras de progreso
import time  # Para agregar pausas y medir tiempo


# 1. Cargamos el archivo CSV que contiene los datos de las flores iris
file_path = "data/iris.csv"  # Ruta al archivo de datos
df = pd.read_csv(file_path)  # Leemos el archivo CSV en un DataFrame

# 2. Separamos los datos en características (X) y etiquetas (y)
X = df.drop("species", axis=1)  # X contiene todas las columnas excepto 'species'
y = df["species"]  # y contiene solo la columna 'species' (lo que queremos predecir)

# 3. Dividimos los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    X, y,  # Datos a dividir
    test_size=0.2,  # 20% para pruebas, 80% para entrenamiento
    random_state=42,  # Semilla para reproducibilidad
    stratify=y  # Mantiene la proporción de clases en ambos conjuntos
)

# 4. Creamos un pipeline que combina el escalado de datos y el modelo
pipeline = Pipeline([
    ('scaler', StandardScaler()),  # Primero normalizamos los datos
    ('rf', RandomForestClassifier(random_state=42))  # Luego aplicamos el clasificador
])

# 5. Definimos los diferentes valores de parámetros que probaremos
param_grid = {
    'rf__n_estimators': [100, 200, 300],  # Número de árboles en el bosque
    'rf__max_depth': [10, 15, 20, None],  # Profundidad máxima de cada árbol
    'rf__min_samples_split': [2, 5],  # Mínimo de muestras para dividir un nodo
    'rf__min_samples_leaf': [1, 2],  # Mínimo de muestras en cada hoja
    'rf__class_weight': ['balanced', None]  # Peso de las clases
}

# 6. Configuramos la búsqueda de los mejores parámetros
grid_search = GridSearchCV(
    pipeline,  # Nuestro pipeline de procesamiento
    param_grid,  # Los parámetros a probar
    cv=5,  # Validación cruzada con 5 pliegues
    scoring='accuracy',  # Métrica para evaluar el rendimiento
    n_jobs=-1,  # Usa todos los núcleos del CPU
    verbose=1  # Muestra progreso
)

# 7. Entrenamos el modelo probando todas las combinaciones de parámetros
grid_search.fit(X_train, y_train)

# 8. Evaluamos el rendimiento del mejor modelo encontrado
best_model = grid_search.best_estimator_  # Obtenemos el mejor modelo
accuracy = best_model.score(X_test, y_test)  # Calculamos la precisión
print(f"Mejores parámetros encontrados: {grid_search.best_params_}")
print(f"Accuracy del modelo: {accuracy:.4f}")

# 9. Guardamos el mejor modelo en un archivo para uso futuro
joblib.dump(best_model, "models/iris_model.pkl")
print("Modelo mejorado guardado en 'models/iris_model.pkl'")

# Función que verifica si ya existe un modelo guardado
def verificar_modelo_existente():
    ruta_modelo = "models/iris_model.pkl"
    return os.path.exists(ruta_modelo)  # Devuelve True si el archivo existe

# Función principal para entrenar el modelo
def entrenar_modelo():
    print("\n🔄 Optimizando modelo con datos actualizados...")
    time.sleep(0.5)  # Pausa breve para mejor experiencia de usuario
    
    # Creamos una barra de progreso para mostrar el avance
    with tqdm(total=4, desc="📊 Preparando modelo", bar_format='{l_bar}{bar:20}{r_bar}') as pbar:
        # Cargamos y preparamos los datos
        df = pd.read_csv("data/iris.csv")
        X = df.drop("species", axis=1)
        y = df["species"]
        pbar.update(1)  # Actualizamos la barra de progreso
        
        # Creamos el pipeline de procesamiento
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('rf', RandomForestClassifier(random_state=42))
        ])
        pbar.update(1)
        
        # Definimos parámetros simplificados para la búsqueda
        param_grid = {
            'rf__n_estimators': [100, 200],
            'rf__max_depth': [10, None]
        }
        pbar.update(1)
        
        # Entrenamos el modelo
        grid_search = GridSearchCV(pipeline, param_grid, cv=5, n_jobs=-1)
        grid_search.fit(X, y)
        pbar.update(1)
    
    # Guardamos el modelo entrenado
    joblib.dump(grid_search.best_estimator_, "models/iris_model.pkl")
    print("\n✨ Modelo actualizado y optimizado exitosamente ✨")

# Función principal que controla el flujo del programa
def main():
    if not verificar_modelo_existente():
        print("\n🚀 Iniciando primera creación del modelo...")
    else:
        print("\n🔄 Actualizando modelo existente...")
    
    entrenar_modelo()

# Punto de entrada del programa
if __name__ == "__main__":
    main()
