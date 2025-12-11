# main.py - Este es el archivo principal que contiene todo el código para clasificar flores iris

# Importamos las bibliotecas necesarias
import joblib  # Para cargar el modelo entrenado
import numpy as np  # Para operaciones numéricas y arrays
import pandas as pd  # Para manipular datos en formato tabular
import matplotlib.pyplot as plt  # Para crear gráficos
import seaborn as sns  # Para gráficos estadísticos más atractivos
from sklearn.datasets import load_iris  # Para cargar el conjunto de datos iris de ejemplo

def cargar_datos_iris():
    # Esta función carga los datos de iris para poder visualizarlos
    # Cargamos el dataset completo de iris que viene con sklearn
    iris = load_iris()
    # Creamos un DataFrame con los datos y columnas apropiadas
    df = pd.DataFrame(data=np.c_[iris['data'], iris['target']], 
                     columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'target'])
    # Convertimos los números de target (0,1,2) a nombres de especies
    df['species'] = df['target'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})
    return df

def visualizar_prediccion(sample_data, predicted_species, df):
    # Esta función crea gráficos para visualizar dónde se ubica nuestra flor en comparación con otras
    
    # Creamos una figura con dos subgráficos lado a lado
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Primer gráfico: Muestra las características del sépalo (longitud vs ancho)
    sns.scatterplot(data=df, x='sepal_length', y='sepal_width', hue='species', ax=ax1)
    # Agregamos nuestra flor como una estrella roja
    ax1.scatter(sample_data[0], sample_data[1], color='red', marker='*', s=200, label='Tu flor')
    ax1.set_title('Características del Sépalo')
    
    # Segundo gráfico: Muestra las características del pétalo (longitud vs ancho)
    sns.scatterplot(data=df, x='petal_length', y='petal_width', hue='species', ax=ax2)
    # Agregamos nuestra flor como una estrella roja
    ax2.scatter(sample_data[2], sample_data[3], color='red', marker='*', s=200, label='Tu flor')
    ax2.set_title('Características del Pétalo')
    
    # Ajustamos el layout y mostramos los gráficos
    plt.tight_layout()
    plt.show()

def predict_iris(sepal_length, sepal_width, petal_length, petal_width):
    # Esta función hace la predicción real usando nuestro modelo entrenado
    
    # Cargamos el modelo previamente entrenado
    model = joblib.load("models/iris_model.pkl")
    # Preparamos los datos de entrada en el formato correcto
    sample = [[sepal_length, sepal_width, petal_length, petal_width]]
    # Hacemos la predicción
    prediction = model.predict(sample)
    # Obtenemos las probabilidades para cada especie
    probabilities = model.predict_proba(sample)[0]
    
    # Definimos un diccionario para mapear nombres de especies a índices
    species_map = {
        'Iris-setosa': 0,
        'Iris-versicolor': 1,
        'Iris-virginica': 2
    }
    # Convertimos la predicción a un índice numérico
    species_index = species_map.get(prediction[0], 0)
    return species_index, probabilities

def mostrar_estadisticas(sample_date,df):
    print("\n===========Estadisticas comparativas=============")
    medidas = df.groupby('species').mean()


    # Para cada especie o flor sacar las caracteristicas comparativas
    caracteristicas = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
    for i, caract in enumerate(caracteristicas):
        print(f"\n{caract.replace('_', ' ').title()} :")
        print(f"\n Valor: {sample_date[i]:.2}cm")
        print("Medidas especies:")
        for especies in ['setosa', 'versicolor', 'virginica']:
            print(f"-- {especies}: {medidas.loc[especies, caract]:.2}cm")

def main():
    print("\nBienvenido a la clasificacion de flores iris")
    sepal_length = float(input("Introduce la longitud del sépalo en cm: "))
    sepal_width = float(input("Introduce la anchura del sapelo en cm: "))
    petal_length = float(input("Introduce la longitud del petalo en cm: "))
    petal_width = float(input("Introduce la anchura del petalo en cm: "))


    # Guardar los datos del usuario 
    sample_date = [sepal_length, sepal_width, petal_length, petal_width]

    # Cargar datos de iris
    df = cargar_datos_iris()
    especies = ['setosa', 'versicolor', 'virginica']

    # inicializar la prediccion
    species_index , probabilites = predict_iris(sepal_length, sepal_width, petal_length, petal_width)

    # Mostrar prediccion de la flor 
    print("\n===========Prediccion=============")
    print(f"La flor predicha es:{especies[species_index]}")
    print("=== Probabilidades de la prediccion ===")
    for esp , prob in zip(especies, probabilites):
        print(f"{esp}: {prob*100:.2f}%")

    mostrar_estadisticas(sample_date, df)
    visualizar_prediccion(sample_date, especies[species_index], df)

if __name__ == '__main__':
     main()




