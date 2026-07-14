# 🏠 Predicción de Precios de Casas en Quito

Proyecto de Inteligencia Artificial que predice el precio estimado de una casa en Quito, Ecuador, usando un modelo de **Regresión Lineal Múltiple** entrenado con un dataset propio. El sistema está dividido en un backend (API) y un frontend (interfaz web), desplegados de forma independiente.

## 🚀 Demo en producción

- **API (backend):** https://ia-p.onrender.com
- **Documentación interactiva de la API:** https://ia-p.onrender.com/docs
- **Aplicación web (frontend):** *agregar aquí el link de Streamlit Cloud*

## 🧠 Descripción del proyecto

El modelo predice el precio de una vivienda a partir de las siguientes características:

- Área (m²)
- Número de pisos
- Habitaciones
- Baños
- Antigüedad (años)
- Garaje (sí/no)
- Sector (Cumbayá, La Carolina, González Suárez, Tumbaco, Centro Norte, Quitumbe, Calderón, Carapungo, Conocoto, Centro Histórico)

El pipeline completo incluye: preprocesamiento de datos (codificación de variables categóricas y normalización opcional), entrenamiento con `scikit-learn`, evaluación con métricas estándar (R², MAE, MSE, RMSE) y exposición del modelo a través de una API REST.

## 🗂️ Estructura del repositorio

```
├── backend/
│   ├── Dockerfile              # Imagen para desplegar en Render
│   ├── api.py                  # Endpoints de FastAPI
│   ├── model_utils.py          # Preprocesamiento de datos
│   ├── train.py                # Entrenamiento del modelo
│   ├── requirements.txt
│   └── data/
│       └── dataset_casas_quito.csv
└── frontend/
    ├── app_streamlit.py        # Interfaz web (Streamlit)
    └── requirements.txt
```

## 🛠️ Tecnologías usadas

| Componente | Tecnología |
|---|---|
| Backend / API | FastAPI + Uvicorn |
| Modelo | scikit-learn (Regresión Lineal) |
| Procesamiento de datos | pandas, numpy |
| Frontend | Streamlit |
| Despliegue backend | Render (Docker) |
| Despliegue frontend | Streamlit Community Cloud |

## ⚙️ Cómo correr el proyecto en local

### Backend

```bash
cd backend
pip install -r requirements.txt
uvicorn api:app --reload
```

La API queda disponible en `http://localhost:8000` y la documentación interactiva en `http://localhost:8000/docs`.

### Frontend

```bash
cd frontend
pip install -r requirements.txt
streamlit run app_streamlit.py
```

Por defecto el frontend apunta a `http://localhost:8000`. Si quieres que apunte a la API en producción en vez de local, define la variable de entorno `API_URL` antes de correrlo:

```bash
export API_URL="https://ia-p.onrender.com"
streamlit run app_streamlit.py
```

## 📡 Endpoints de la API

| Método | Endpoint | Descripción |
|---|---|---|
| `GET` | `/` | Verifica que la API esté activa |
| `POST` | `/entrenar` | Entrena (o re-entrena) el modelo con el dataset |
| `GET` | `/metricas` | Devuelve las métricas del último entrenamiento |
| `POST` | `/predecir` | Devuelve el precio estimado de una casa |

⚠️ **Importante:** el modelo debe entrenarse al menos una vez llamando a `POST /entrenar` antes de poder usar `/predecir`. En el plan gratuito de Render el disco no es permanente, así que si el servicio se reinicia por inactividad, hay que volver a entrenar.

## 🧪 Probar la API con Postman (opcional)

No es obligatorio usar Postman — el frontend en Streamlit ya permite entrenar y predecir sin necesidad de herramientas extra — pero si el equipo quiere probar la API directamente, aquí están las peticiones:

**1. Verificar que la API esté activa**
- Método: `GET`
- URL: `https://ia-p.onrender.com/`

**2. Entrenar el modelo**
- Método: `POST`
- URL: `https://ia-p.onrender.com/entrenar`
- Body → raw → JSON:
```json
{
  "test_size": 0.2,
  "normalizar": true,
  "random_state": 42
}
```

**3. Consultar métricas**
- Método: `GET`
- URL: `https://ia-p.onrender.com/metricas`

**4. Predecir el precio de una casa**
- Método: `POST`
- URL: `https://ia-p.onrender.com/predecir`
- Body → raw → JSON:
```json
{
  "area_m2": 120,
  "pisos": 2,
  "habitaciones": 3,
  "banos": 2,
  "antiguedad": 5,
  "garaje": 1,
  "sector": "Cumbaya"
}
```

En Postman, recuerda seleccionar **Body → raw → JSON** (no "Text"), para que la API reciba el `Content-Type: application/json` correctamente.

## 📊 Métricas del modelo

Las métricas (R², MAE, MSE, RMSE) se generan dinámicamente en cada entrenamiento y se pueden consultar desde:
- La pestaña **"Métricas del modelo"** en el frontend, o
- El endpoint `GET /metricas` de la API.

## 👥 Integrantes del equipo

- Kyara Altamirano
- Santiago Vargas
- Patiño Josue

## 📄 Licencia

Proyecto académico con fines educativos.
