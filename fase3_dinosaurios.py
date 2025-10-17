import numpy as np
import matplotlib.pyplot as plt
from fase2_dinosaurios import train_svm_dinosaurios
import pandas as pd

def plot_svm_analysis(X, y, w, b):
    """
    Visualiza la frontera de decisión, márgenes y vectores soporte del SVM
    """
    # Diagnóstico inicial
    print("\nDiagnóstico de la visualización:")
    print(f"Dimensiones de X: {X.shape}")
    print(f"Dimensiones de y: {y.shape}")
    print(f"Valores únicos en y: {np.unique(y)}")
    print(f"Vector w: {w}")
    print(f"Bias b: {b}")
    plt.figure(figsize=(12, 8))
    
    # Crear malla para visualización
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                        np.linspace(y_min, y_max, 100))
    
    # Calcular valores de decisión
    Z = np.dot(np.c_[xx.ravel(), yy.ravel()], w) + b
    Z = Z.reshape(xx.shape)
    
    # Graficar regiones de decisión
    contours = plt.contour(xx, yy, Z, colors='k', levels=[-1, 0, 1],
                          linestyles=['--', '-', '--'])
    
    # Añadir líneas a la leyenda manualmente
    plt.plot([], [], 'k-', label='Frontera de decisión (w^T x + b = 0)')
    plt.plot([], [], 'k--', label='Márgenes (w^T x + b = ±1)')
    
    # Calcular márgenes para cada punto
    margins = y * (np.dot(X, w) + b)
    support_vectors = np.abs(margins - 1) < 0.1  # Tolerancia numérica
    
    # Graficar puntos
    plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], c='b', marker='o',
               label='Clase +1')
    plt.scatter(X[y == -1][:, 0], X[y == -1][:, 1], c='r', marker='x',
               label='Clase -1')
    
    # Resaltar vectores soporte
    plt.scatter(X[support_vectors][:, 0], X[support_vectors][:, 1],
               s=200, facecolors='none', edgecolors='g',
               label='Vectores Soporte')
    
    # Anotaciones y formato
    margin_width = 2 / np.linalg.norm(w)
    plt.title(f'Análisis SVM: Frontera, Márgenes y Vectores Soporte\n' +
             f'Ancho del margen: {margin_width:.3f}')
    plt.xlabel('Característica 1')
    plt.ylabel('Característica 2')
    plt.legend()
    plt.grid(True)
    
    # Guardar figura
    plt.savefig('svm_analysis.png')
    plt.close()
    
    return margin_width, np.sum(support_vectors)

if __name__ == "__main__":
    # Cargar datos
    try:
        df = pd.read_csv("train_linear.csv")  # Quitamos "data/" del path
        X = df[['x1','x2']].to_numpy()
        y = df['y'].to_numpy()
        
        # Diagnóstico de las etiquetas originales
        print("Valores únicos en y antes de conversión:", np.unique(y))
        print("Distribución de clases original:")
        for val in np.unique(y):
            print(f"Clase {val}: {np.sum(y == val)} muestras")
        
        # Conversión de etiquetas asegurando ambas clases
        if set(np.unique(y)) == {0, 1}:
            y = np.where(y == 0, -1, 1)  # Convertir 0,1 a -1,1
        elif set(np.unique(y)) == {1, 2}:
            y = np.where(y == 1, -1, 1)  # Convertir 1,2 a -1,1
        
        # Verificar conversión
        print("\nValores únicos en y después de conversión:", np.unique(y))
        print("Distribución de clases después de conversión:")
        for val in np.unique(y):
            print(f"Clase {val}: {np.sum(y == val)} muestras")
            
    except FileNotFoundError:
        print("Generando datos sintéticos balanceados...")
        np.random.seed(42)
        n_samples_per_class = 50  # 50 muestras por clase
        
        # Generar datos para clase -1
        X1 = np.random.randn(n_samples_per_class, 2) - np.array([2, 2])
        y1 = np.ones(n_samples_per_class) * -1
        
        # Generar datos para clase +1
        X2 = np.random.randn(n_samples_per_class, 2) + np.array([2, 2])
        y2 = np.ones(n_samples_per_class)
        
        # Combinar datos
        X = np.vstack([X1, X2])
        y = np.hstack([y1, y2])
    
    # Entrenar modelo
    C = 1.0
    w, b, _ = train_svm_dinosaurios(X, y, C=C)
    
    # Analizar y visualizar resultados
    margin_width, n_support = plot_svm_analysis(X, y, w, b)
    
    # Guardar análisis
    with open('svm_analysis.txt', 'w') as f:
        f.write("Análisis del SVM\n")
        f.write("================\n\n")
        f.write(f"Ancho del margen: {margin_width:.3f}\n")
        f.write(f"Número de vectores soporte: {n_support}\n")
        f.write(f"Norma del vector w: {np.linalg.norm(w):.3f}\n")
        f.write(f"Bias b: {b:.3f}\n")
        f.write("\nInterpretación:\n")
        f.write("- El ancho del margen es 2/|w|\n")
        f.write("- Los vectores soporte son los puntos que definen el margen\n")
        f.write("- Modificar b desplaza la frontera sin cambiar su orientación\n")