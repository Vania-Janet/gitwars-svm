import numpy as np

def hinge_loss(w, b, X, y, C):
    '''
    Calcula la pérdida hinge regularizada del SVM lineal.
    L(w,b) = 0.5 * ||w||^2 + C * sum_i max(0, 1 - y_i * (w^T x_i + b))

    Parámetros:
      w: array-like de shape (d,)
      b: float (sesgo)
      X: array-like de shape (n, d) o (d,)  -> si viene (d,), se interpreta como 1 muestra
      y: array-like de shape (n,) o escalar en {-1, +1}
      C: float >= 0

    Devuelve:
      float: valor escalar de la función objetivo
    '''
    # ---- Normalización de tipos y shapes ----
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 1:               # caso de un solo ejemplo
        X = X.reshape(1, -1)

    w = np.asarray(w, dtype=np.float64).reshape(-1)
    b = float(b)

    y = np.asarray(y, dtype=np.float64).reshape(-1)
    n, d = X.shape

    # ---- Validaciones de consistencia ----
    if w.shape[0] != d:
        raise ValueError(f"Dim mismatch: w tiene {w.shape[0]} y X tiene {d} features.")
    if y.size == 1 and n > 1:
        # Permitir etiqueta escalar repetida para todo el batch
        y = np.full(n, float(y.item()), dtype=np.float64)
    if y.shape[0] != n:
        raise ValueError(f"Dim mismatch: y tiene {y.shape[0]} y X tiene {n} muestras.")
    if not np.all(np.isin(np.unique(y), (-1.0, 1.0))):
        raise ValueError("y debe estar en {-1, +1}.")
    if C < 0:
        raise ValueError("C debe ser no negativo (recomendado C > 0).")

    # ---- Cálculo vectorizado de la pérdida ----
    margins = y * (X @ w + b)           # y_i * (w^T x_i + b)
    hinge = np.maximum(0.0, 1.0 - margins)
    reg = 0.5 * np.dot(w, w)            # 0.5 ||w||^2
    data = C * np.sum(hinge)            # C * sum_i hinge_i
    L = reg + data

    # Devolver como escalar Python para máxima compatibilidad
    return float(L)

import numpy as np
import pandas as pd

# Usa el original o el limpio; ambos tienen y en {-1,+1}
df = pd.read_csv("data/train_linear.csv")  # o "data/train_linear_clean.csv" si quieres
X = df[["x1", "x2"]].to_numpy(dtype=float)
y = df["y"].to_numpy(dtype=float)  # etiquetas ya en {-1,+1]

print("Shapes:", X.shape, y.shape)
print("Valores únicos de y:", np.unique(y))
