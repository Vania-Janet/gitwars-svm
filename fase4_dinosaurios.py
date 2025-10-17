import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
import pandas as pd

def rbf_kernel(X1, X2, gamma):
    """
    Implementa el kernel RBF: K(x,z) = exp(-gamma ||x-z||^2)
    """
    dist_matrix = np.sum(X1**2, axis=1).reshape(-1, 1) + np.sum(X2**2, axis=1) - 2 * np.dot(X1, X2.T)
    return np.exp(-gamma * dist_matrix)

def plot_svm_rbf_decision_boundary(X, y, svm_model, gamma, C, title=None):
    """
    Visualiza la frontera de decisión del SVM con kernel RBF
    """
    plt.figure(figsize=(12, 8))
    
    # Crear malla para visualización
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                        np.linspace(y_min, y_max, 200))
    
    # Evaluar el modelo en la malla
    Z = svm_model.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Graficar contornos de decisión
    plt.contour(xx, yy, Z, colors='k', levels=[-1, 0, 1],
                linestyles=['--', '-', '--'])
    plt.plot([], [], 'k-', label='Frontera de decisión')
    plt.plot([], [], 'k--', label='Márgenes')
    
    # Graficar puntos de datos
    plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], c='b', marker='o',
               label='Clase +1')
    plt.scatter(X[y == -1][:, 0], X[y == -1][:, 1], c='r', marker='x',
               label='Clase -1')
    
    # Resaltar vectores soporte
    support_vectors = svm_model.support_vectors_
    plt.scatter(support_vectors[:, 0], support_vectors[:, 1],
               s=200, facecolors='none', edgecolors='g',
               label='Vectores Soporte')
    
    # Configuración del gráfico
    if title is None:
        title = f'SVM con Kernel RBF (γ={gamma}, C={C})\n'
        title += f'Vectores Soporte: {len(support_vectors)}'
    plt.title(title)
    plt.xlabel('Característica 1')
    plt.ylabel('Característica 2')
    plt.legend()
    plt.grid(True)
    
    return plt

def analyze_hyperparameters(X, y, gammas=[0.1, 1, 10], Cs=[0.1, 1, 10]):
    """
    Analiza el efecto de diferentes valores de γ y C
    """
    fig, axes = plt.subplots(len(gammas), len(Cs), figsize=(15, 15))
    
    for i, gamma in enumerate(gammas):
        for j, C in enumerate(Cs):
            # Entrenar modelo
            svm = SVC(kernel='rbf', gamma=gamma, C=C)
            svm.fit(X, y)
            
            # Crear subplot
            plt.sca(axes[i, j])
            plot_svm_rbf_decision_boundary(X, y, svm, gamma, C,
                                        f'γ={gamma}, C={C}')
            
    plt.tight_layout()
    plt.savefig('svm_rbf_analysis.png')
    plt.close()

if __name__ == "__main__":
    # Cargar datos no lineales
    try:
        df = pd.read_csv("train_nonlinear.csv")
        X = df[['x1','x2']].to_numpy()
        y = df['y'].to_numpy()
        y = np.where(y == 0, -1, 1)  # Convertir a {-1, 1}
    except FileNotFoundError:
        print("Generando datos sintéticos no lineales...")
        np.random.seed(42)
        n_samples = 100
        
        # Generar datos en forma de círculos concéntricos
        radius1, radius2 = 2, 4
        n_samples_per_class = n_samples // 2
        
        # Clase interior (-1)
        theta = np.random.uniform(0, 2*np.pi, n_samples_per_class)
        r = np.random.normal(radius1, 0.4, n_samples_per_class)
        X1 = np.column_stack([r*np.cos(theta), r*np.sin(theta)])
        y1 = np.ones(n_samples_per_class) * -1
        
        # Clase exterior (+1)
        theta = np.random.uniform(0, 2*np.pi, n_samples_per_class)
        r = np.random.normal(radius2, 0.4, n_samples_per_class)
        X2 = np.column_stack([r*np.cos(theta), r*np.sin(theta)])
        y2 = np.ones(n_samples_per_class)
        
        X = np.vstack([X1, X2])
        y = np.hstack([y1, y2])
    
    # Analizar diferentes combinaciones de hiperparámetros
    gammas = [0.1, 1, 10]
    Cs = [0.1, 1, 10]
    analyze_hyperparameters(X, y, gammas, Cs)
    
    # Entrenar y visualizar un modelo específico
    best_gamma, best_C = 1.0, 1.0
    svm = SVC(kernel='rbf', gamma=best_gamma, C=best_C)
    svm.fit(X, y)
    
    # Visualizar el mejor modelo
    plt = plot_svm_rbf_decision_boundary(X, y, svm, best_gamma, best_C)
    plt.savefig('svm_rbf_best.png')
    plt.close()
    
    # Guardar análisis
    with open('svm_rbf_analysis.txt', 'w') as f:
        f.write("Análisis del SVM con Kernel RBF\n")
        f.write("============================\n\n")
        f.write(f"Mejores hiperparámetros:\n")
        f.write(f"- gamma: {best_gamma}\n")
        f.write(f"- C: {best_C}\n\n")
        f.write(f"Número de vectores soporte: {len(svm.support_vectors_)}\n")
        f.write(f"Accuracy: {svm.score(X, y):.4f}\n\n")
        f.write("Observaciones:\n")
        f.write("- Un gamma mayor produce fronteras más flexibles\n")
        f.write("- Un C mayor permite menos errores de clasificación\n")
        f.write("- La interacción entre gamma y C afecta el equilibrio\n")
        f.write("  entre bias y varianza del modelo")