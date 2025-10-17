# TODO: Implementar la pérdida hinge y su gradiente
def hinge_loss(w, b, X, y, C):
    '''
    Calcula la pérdida hinge regularizada del SVM lineal.
    L = 0.5 * |w|^2 + C * sum(max(0, 1 - y_i * (w^T * x_i + b)))
    '''
    import numpy as np
    
    # Término de regularización: 0.5 * |w|^2
    regularization = 0.5 * np.dot(w, w)
    
    # Predicciones: w^T * x_i + b
    predictions = np.dot(X, w) + b
    
    # Margen: y_i * (w^T * x_i + b)
    margins = y * predictions
    
    # Hinge loss: max(0, 1 - margen)
    hinge = np.maximum(0, 1 - margins)
    
    # Pérdida total
    L = regularization + C * np.sum(hinge)
    
    return L


# ===== VALIDACIÓN =====
if __name__ == "__main__":
    import numpy as np
    
    # Test 1: Lote pequeño (2 muestras)
    print("Test 1: Lote pequeño (2 muestras)")
    w = np.array([1.0, -0.5])
    b = 0.0
    X = np.array([[1.0, 2.0], [-1.0, 1.0]])
    y = np.array([1, -1])
    C = 1.0
    
    loss = hinge_loss(w, b, X, y, C)
    print(f"   Loss = {loss}")
    print(f"   Tipo: {type(loss).__name__} ✓")
    assert isinstance(loss, (float, np.floating)), "Debe devolver escalar"
    print()
    
    # Test 2: Lote grande (1000 muestras)
    print("Test 2: Lote grande (1000 muestras)")
    X_big = np.random.randn(1000, 10)
    y_big = np.random.choice([-1, 1], size=1000)
    w_big = np.random.randn(10)
    b_big = 0.5
    C = 0.1
    
    loss_big = hinge_loss(w_big, b_big, X_big, y_big, C)
    print(f"   Loss = {loss_big:.4f}")
    print(f"   Tipo: {type(loss_big).__name__} ✓")
    assert isinstance(loss_big, (float, np.floating)), "Debe devolver escalar"
    print()
    
    # Test 3: Una sola muestra
    print("Test 3: Una sola muestra")
    X_single = np.array([[1.0, 2.0]])
    y_single = np.array([1])
    loss_single = hinge_loss(w, b, X_single, y_single, C)
    print(f"   Loss = {loss_single}")
    print(f"   Tipo: {type(loss_single).__name__} ✓")
    assert isinstance(loss_single, (float, np.floating)), "Debe devolver escalar"
    print()
 
    # Test 5: Tipos mixtos (int y float)
    print("Test 5: Tipos mixtos")
    X_mixed = np.array([[1, 2], [3, 4]])  # int
    y_mixed = np.array([1.0, -1.0])  # float
    w_mixed = np.array([0.5, 0.5])
    b_mixed = 0
    C = 1
    
    loss_mixed = hinge_loss(w_mixed, b_mixed, X_mixed, y_mixed, C)
    print(f"   Loss = {loss_mixed}")
    print(f"   Tipo: {type(loss_mixed).__name__} ✓")
    assert isinstance(loss_mixed, (float, np.floating)), "Debe devolver escalar"
    print()
