import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import json, pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from model_utils import preprocesar, construir_fila_nueva, COLUMNAS_NUMERICAS

DATASET_PATH  = "proyecto/data/dataset_casas_quito.csv"
MODELO_PATH   = "proyecto/models/modelo_regresion.pkl"
METRICAS_PATH = "proyecto/models/metricas.json"
GRAFICA_PATH  = "proyecto/models/grafica_real_vs_predicho.png"

def entrenar(test_size=0.2, normalizar=True, random_state=42):
    # 1. Cargar datos
    datos = pd.read_csv(DATASET_PATH)
    print("Dataset cargado:", datos.shape)
    print(datos.head())
    print(datos.describe())

    # 2. Limpieza
    datos = datos.dropna().drop_duplicates()
    p1, p99 = datos["precio"].quantile([0.01, 0.99])
    datos = datos[(datos["precio"] >= p1) & (datos["precio"] <= p99)]

    # 3. Separar caracteristicas (X) y etiqueta (y)
    X, y = preprocesar(datos, entrenando=True)

    # 4. Separar en entrenamiento y prueba
    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    print(f"Entrenamiento: {len(x_train)} | Prueba: {len(x_test)}")

    # 5. Normalizar
    scaler = None
    if normalizar:
        scaler = StandardScaler()
        x_train[COLUMNAS_NUMERICAS] = scaler.fit_transform(x_train[COLUMNAS_NUMERICAS])
        x_test[COLUMNAS_NUMERICAS]  = scaler.transform(x_test[COLUMNAS_NUMERICAS])

    # 6. Crear modelo
    modelo = LinearRegression()

    # 7. Entrenar
    modelo.fit(x_train, y_train)

    # 8. Predecir con datos de prueba
    y_predichas = modelo.predict(x_test)

    # 9. Metricas de evaluacion
    mse  = mean_squared_error(y_test, y_predichas)
    r2   = r2_score(y_test, y_predichas)
    mae  = mean_absolute_error(y_test, y_predichas)
    rmse = np.sqrt(mse)

    print("\n===== METRICAS DE EVALUACION =====")
    print(f"MSE:  {mse:,.2f}")
    print(f"RMSE: {rmse:,.2f}")
    print(f"MAE:  {mae:,.2f}")
    print(f"El coeficiente de determinacion (R2) es {r2:.4f}")

    # 10. Coeficientes e intercepto
    print("\nCoeficientes: ")
    for nombre, coef in zip(x_train.columns, modelo.coef_):
        print(f"  {nombre}: {coef:.2f}")
    print(f"Intercepto: {modelo.intercept_:.2f}")

    # 11. Prediccion con dato nuevo
    print("\n===== PREDICCION CON DATO NUEVO =====")
    print("(Casa 120m2, 2 pisos, 3 hab, 2 baños, 5 años, garaje, La Carolina)")
    fila = construir_fila_nueva(120, 2, 3, 2, 5, 1, "La Carolina")
    X_nuevo = preprocesar(fila, entrenando=False)
    if scaler:
        X_nuevo[COLUMNAS_NUMERICAS] = scaler.transform(X_nuevo[COLUMNAS_NUMERICAS])
    precio_estimado = modelo.predict(X_nuevo)[0]
    print(f"El precio estimado es: ${precio_estimado:,.2f} USD")

    # 12. Grafica  (adaptacion de la grafica de clase a regresion multiple)
    # En clase con 1 variable: scatter(x,y) + plot(x_test, y_predichas)
    # Con varias variables: graficamos Precio real vs Precio predicho
    plt.figure(figsize=(7,6))
    plt.scatter(y_test, y_predichas, alpha=0.5, label="Predicciones")
    minv = min(y_test.min(), y_predichas.min())
    maxv = max(y_test.max(), y_predichas.max())
    plt.plot([minv, maxv], [minv, maxv], color="red", label="Prediccion perfecta")
    plt.title("Regresion Lineal Multiple\nPrecio real vs Precio predicho")
    plt.xlabel("Precio real (USD)")
    plt.ylabel("Precio predicho (USD)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(GRAFICA_PATH)
    plt.show()

    # 13. Guardar modelo
    metricas = {
        "MAE": round(mae,2), "RMSE": round(rmse,2),
        "MSE": round(mse,2), "R2": round(r2,4),
        "n_entrenamiento": len(x_train), "n_prueba": len(x_test),
        "parametros": {"test_size": test_size, "normalizar": normalizar,
                    "random_state": random_state},
    }
    with open(MODELO_PATH, "wb") as f:
        pickle.dump({"modelo": modelo, "scaler": scaler,
                    "columnas": list(x_train.columns), "normalizar": normalizar}, f)
    with open(METRICAS_PATH, "w") as f:
        json.dump(metricas, f, indent=2)

    print("\n✅ Modelo guardado correctamente")
    return metricas

if __name__ == "__main__":
    entrenar()
