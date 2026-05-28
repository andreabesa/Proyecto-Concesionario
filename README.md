# 🚗 Predictor de Precios de Vehículos

Proyecto de Machine Learning para predecir el precio de vehículos de segunda mano a partir de sus características, utilizando un modelo de **Random Forest Regressor** entrenado sobre un dataset de anuncios de coches.

---

## 📋 Descripción

A partir de un dataset de vehículos (`autos.csv`) con más de 370.000 registros, se realiza un análisis exploratorio, limpieza de datos, ingeniería de variables y entrenamiento de un modelo de regresión capaz de estimar el precio de un coche dado un conjunto de características como la marca, el modelo, el kilometraje, el tipo de combustible o el año de matriculación.

---

## 📁 Estructura del proyecto

```
Proyecto/
│
├── csv/
│   └── autos.csv               # Dataset original
│
├── model/
│   ├── modelo.pkl              # Modelo RandomForest entrenado
│   ├── columnas.pkl            # Lista de columnas usadas en el entrenamiento
│   ├── freq_brand.pkl          # Frecuencias de marcas (frequency encoding)
│   └── freq_model.pkl          # Frecuencias de modelos (frequency encoding)
│
├── marcas_modelos.json         # Diccionario marca → lista de modelos disponibles
├── notebook.ipynb              # Notebook principal de análisis y entrenamiento
├── predecir.py                 # Lógica de predicción
├── app.py                      # Aplicación Flask (API REST)
├── predictor-precios.html      # Frontend de la aplicación
└── requirements.txt            # Dependencias del proyecto
```

---

## 📊 Dataset

El dataset contiene **21 columnas** con las siguientes categorías:

| Tipo | Columnas |
|---|---|
| Numéricas continuas | `price`, `powerPS`, `kilometer`, `yearOfRegistration`, `postalCode` |
| Categóricas nominales | `brand`, `model`, `vehicleType`, `gearbox`, `fuelType`, `notRepairedDamage`, `seller`, `offerType` |
| Temporales | `dateCrawled`, `dateCreated`, `lastSeen`, `monthOfRegistration` |

**Estadísticas clave del dataset limpio:**
- Precio medio: ~17.295 €
- Potencia media: ~115 CV
- Kilometraje medio: ~125.618 km
- Marca más frecuente: Volkswagen
- Tipo de marcha predominante: Manual
- Combustible predominante: Gasolina

---

## 🧹 Limpieza de datos

- Eliminación de filas con nulos en columnas clave: `gearbox`, `vehicleType`, `model`, `fuelType`, `notRepairedDamage`
- Filtrado de outliers en precio: entre **500 € y 30.000 €**
- Filtrado de outliers en año de matriculación: entre **1900 y 2026**
- Sustitución de mes de registro `0` por `1`
- Eliminación de la columna `nrOfPictures` (sin variabilidad)
- Creación de la variable `antiguedad` = 2026 - `yearOfRegistration`

---

## ⚙️ Preprocesamiento y features

- **Frequency Encoding** para `brand` y `model` (número de apariciones en el dataset)
- **One-Hot Encoding** para `notRepairedDamage`, `vehicleType` y `fuelType`
- Eliminación de columnas no informativas: `dateCrawled`, `lastSeen`, `postalCode`, `index`, `name`, `seller`, `offerType`, `abtest`, `gearbox`
- Extracción de `yearOfPurchase` y `monthOfPurchase` desde `dateCreated`
- **Transformación logarítmica** del target (`price`) con `np.log1p` para mejorar la distribución y reducir el error

---

## 🤖 Modelo

Se utiliza un **Random Forest Regressor** con los siguientes hiperparámetros finales:

```python
RandomForestRegressor(
    n_estimators=50,
    max_depth=12,
    max_features='sqrt',
    min_samples_split=2,
    random_state=42,
    n_jobs=-1
)
```

El target se transforma con `log1p` antes del entrenamiento y las predicciones se revierten con `expm1`.

### Resultados del modelo

| Métrica | Valor |
|---|---|
| MAE | ~1.088 € |
| R² | > 0.85 |
| MAPE | < 20% |

**Error por rango de precio (datos limpios):**

| Rango | MAE |
|---|---|
| 500 € – 1.000 € | ~1 € |
| 1.000 € – 5.000 € | ~1 € |
| 5.000 € – 15.000 € | ~13 € |
| 15.000 € – 30.000 € | ~111 € |

También se exploró **GridSearchCV** para optimización de hiperparámetros con scoring `r2`.

---

## 📉 Análisis adicional

### KMeans Clustering
Se aplica KMeans sobre una muestra de 30.000 registros para identificar agrupaciones naturales en los datos. El número óptimo de clusters determinado por el **Método del Codo** y el **Coeficiente de Silueta** es **k = 4**.

### PCA
Se aplica PCA para reducir la dimensionalidad del dataset (de 27 columnas a 2 componentes principales). Se verifica que se necesitan **21 componentes** para conservar el 95% de la varianza. Para visualización e input de KMeans se usan **2 componentes**.

---

## 🚀 Instalación y uso

### 1. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 2. Ejecutar la aplicación Flask

```bash
python app.py
```

### 3. Abrir el frontend

Abre `predictor-precios.html` en el navegador o accede a `http://localhost:5000`.

---

## 🛠️ Tecnologías

- **Python 3.x**
- `pandas`, `numpy` — manipulación de datos
- `scikit-learn` — modelo, preprocesamiento, clustering, PCA
- `seaborn`, `matplotlib` — visualización
- `joblib` — serialización del modelo
- `Flask` + `flask-cors` — API REST
