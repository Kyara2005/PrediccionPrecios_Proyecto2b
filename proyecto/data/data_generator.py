import numpy as np
import pandas as pd

np.random.seed(42)

# Precio por m2 (USD) por sector — calibrado con datos Properati / Plusvalia
SECTORES = {
    "Cumbaya":          (1300, 1900),
    "La Carolina":      (1200, 1700),
    "Gonzalez Suarez":  (1400, 2000),
    "Tumbaco":          (900,  1300),
    "Centro Norte":     (1000, 1400),
    "Quitumbe":         (550,  800),
    "Calderon":         (500,  750),
    "Carapungo":        (480,  700),
    "Conocoto":         (700,  1000),
    "Centro Historico": (650,  950),
}

def generar_dataset(n=20000):
    filas = []
    sectores = list(SECTORES.keys())
    for _ in range(n):
        sector = np.random.choice(sectores)
        precio_min_m2, precio_max_m2 = SECTORES[sector]
        precio_m2_base = np.random.uniform(precio_min_m2, precio_max_m2)
        area_m2      = float(np.clip(np.random.normal(140, 50), 45, 400))
        pisos        = np.random.choice([1, 2, 3], p=[0.35, 0.5, 0.15])
        habitaciones = np.random.choice([1, 2, 3, 4, 5], p=[0.05, 0.25, 0.4, 0.2, 0.1])
        banos        = np.random.choice([1, 2, 3, 4], p=[0.2, 0.45, 0.25, 0.1])
        antiguedad   = int(np.clip(np.random.exponential(8), 0, 45))
        garaje       = np.random.choice([0, 1], p=[0.2, 0.8])
        precio  = area_m2 * precio_m2_base
        precio *= 1 + 0.03 * (pisos - 1)
        precio *= 1 + 0.02 * max(0, habitaciones - 2)
        precio *= 1 + 0.015 * max(0, banos - 1)
        precio *= 1 - 0.004 * antiguedad
        precio *= 1 + (0.04 if garaje == 1 else 0)
        precio  = precio * np.random.normal(1.0, 0.08)
        precio  = round(max(precio, 15000), 2)
        filas.append({"sector": sector, "area_m2": round(area_m2,1),
                      "pisos": int(pisos), "habitaciones": int(habitaciones),
                      "banos": int(banos), "antiguedad": antiguedad,
                      "garaje": int(garaje), "precio": precio})
    return pd.DataFrame(filas)

if __name__ == "__main__":
    df = generar_dataset()
    df.to_csv("proyecto/data/dataset_casas_quito.csv", index=False)
    print(f"Dataset generado: {len(df)} registros")
    print(df.head())
    print(df.describe())
