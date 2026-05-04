# 🍷 Wine Quality ML Pipeline

Pipeline de Machine Learning completamente automatizado con **GitHub Actions** y **MLflow**.
Clasifica vinos tintos como *buena calidad* o *baja calidad* a partir de sus
características fisicoquímicas, aplicando prácticas modernas de MLOps.

---

## 🎯 Problema que se resuelve

La industria vitivinícola necesita formas objetivas y reproducibles de evaluar la calidad
de sus vinos. Tradicionalmente, esta evaluación depende de catadores expertos, lo que la
hace subjetiva, costosa y difícil de escalar.

Este proyecto propone un modelo de machine learning que predice si un vino tinto es de
**buena calidad** o **baja calidad** basándose exclusivamente en sus propiedades
fisicoquímicas medibles en laboratorio, como la acidez, el pH, el contenido de alcohol
y otros indicadores. El objetivo es construir un sistema reproducible, trazable y
automatizado que pueda ser auditado en cualquier momento.

---

## 🗂️ Dataset

**Wine Quality (Red) — UCI Machine Learning Repository**

| Atributo     | Detalle                                                                 |
|--------------|-------------------------------------------------------------------------|
| Nombre       | Wine Quality (Red)                                                      |
| Fuente       | UCI Machine Learning Repository                                         |
| Registros    | 1,599 muestras de vino tinto                                            |
| Features     | 11 variables fisicoquímicas continuas                                   |
| Variable objetivo | Calidad del vino (score del 3 al 8)                               |
| Formato      | CSV separado por punto y coma (`;`)                                     |
| Licencia     | Pública y libre de uso                                                  |
| URL          | https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/ |

### Variables del dataset

| Variable                  | Descripción                                              |
|---------------------------|----------------------------------------------------------|
| `fixed acidity`           | Acidez fija del vino                                     |
| `volatile acidity`        | Acidez volátil (vinagre) — valores altos reducen calidad |
| `citric acid`             | Ácido cítrico — aporta frescura al vino                  |
| `residual sugar`          | Azúcar residual tras la fermentación                     |
| `chlorides`               | Contenido de sal                                         |
| `free sulfur dioxide`     | SO₂ libre — previene oxidación y crecimiento microbiano  |
| `total sulfur dioxide`    | SO₂ total (libre + ligado)                               |
| `density`                 | Densidad del vino                                        |
| `pH`                      | Nivel de acidez general                                  |
| `sulphates`               | Sulfatos — contribuyen al aroma y conservación           |
| `alcohol`                 | Porcentaje de alcohol por volumen                        |
| `quality`                 | Puntuación sensorial del vino (3 a 8) — variable original|

### Transformación del target

La variable `quality` original es un score numérico. Para este proyecto se transforma
en una **clasificación binaria**:

- `quality >= 7` → **1** (buena calidad)
- `quality < 7`  → **0** (baja calidad)

Esto permite usar métricas de clasificación claras como Accuracy y F1-Score, y construir
un modelo más interpretable para el negocio.

### ¿Por qué este dataset?

- Es una fuente externa reconocida y ampliamente usada en investigación
- No proviene de `sklearn.datasets`
- Tiene datos reales de laboratorio, no sintéticos
- Permite construir un problema de negocio concreto y explicable
- Es suficientemente pequeño para entrenar rápido y suficientemente rico para demostrar preprocesamiento

---

## 🏗️ Arquitectura del proyecto

wine-ml-pipeline/
├── .github/
│ └── workflows/
│ └── ml.yml # Pipeline de GitHub Actions (CI/CD)
├── config/
│ └── config.yaml # Hiperparámetros, rutas y umbrales (sin hardcodear)
├── src/
│ └── train.py # Script principal: carga → preprocesa → entrena → registra
├── tests/
│ └── test_model.py # Pruebas de validación del modelo y los datos
├── Makefile # Comandos centralizados para local y CI/CD
├── requirements.txt # Dependencias del proyecto
└── README.md # Documentación del proyecto


---

## ⚙️ Tecnologías utilizadas

| Tecnología     | Versión | Uso                                      |
|----------------|---------|------------------------------------------|
| Python         | 3.10    | Lenguaje principal                       |
| LightGBM       | 4.3.0   | Modelo de clasificación                  |
| scikit-learn   | 1.4.2   | Pipeline, escalamiento y métricas        |
| MLflow         | 2.13.0  | Tracking y registro del modelo           |
| pandas         | 2.2.2   | Carga y manipulación de datos            |
| PyYAML         | 6.0.1   | Lectura del archivo de configuración     |
| joblib         | 1.4.2   | Serialización del modelo a .pkl          |
| GitHub Actions | —       | Automatización CI/CD en la nube          |

---

## 🔧 Preprocesamiento aplicado

El script `src/train.py` aplica los siguientes pasos antes de entrenar:

1. **Carga del dataset** desde la URL de UCI con separador `;`
2. **Manejo de nulos** — se eliminan filas con valores faltantes (el dataset no tiene, pero se valida)
3. **Creación del target binario** — `quality >= 7` → 1, `quality < 7` → 0
4. **División train/test** — 80% entrenamiento, 20% prueba, con estratificación para preservar proporciones
5. **Escalamiento** — `StandardScaler` aplicado dentro del pipeline de scikit-learn

---

## 🤖 Modelo

Se utiliza un **LightGBM Classifier** envuelto en un pipeline de scikit-learn junto
con `StandardScaler`. LightGBM fue elegido porque:

- Es rápido y eficiente en datos tabulares
- Maneja bien desbalance de clases
- Se integra nativamente con pipelines de scikit-learn y con MLflow
- Permite registrar parámetros fácilmente

Los hiperparámetros se definen en `config/config.yaml`:

```yaml
model:
  n_estimators: 200
  max_depth: 6
  learning_rate: 0.05
  num_leaves: 31
  random_state: 42
```

---

## 📊 Métricas y umbrales de calidad

| Métrica    | Umbral mínimo | Resultado esperado |
|------------|---------------|--------------------|
| Accuracy   | ≥ 0.70        | ~0.83              |
| F1-Score   | ≥ 0.68        | ~0.80              |

Los umbrales están definidos en `config/config.yaml` y son validados automáticamente
por `tests/test_model.py` dentro del pipeline de CI/CD. Si el modelo no supera los
umbrales, el pipeline falla y no se publican artefactos.

---

## 📝 Tracking con MLflow

El script registra los siguientes elementos en cada ejecución:

| Elemento        | Qué se registra                                                    |
|-----------------|--------------------------------------------------------------------|
| Parámetros      | n_estimators, max_depth, learning_rate, num_leaves, dataset, test_size |
| Métricas        | accuracy, f1_score                                                 |
| Firma           | Generada con `infer_signature` — define tipos de entrada y salida  |
| Input example   | 5 filas reales del conjunto de prueba                              |
| Modelo          | Guardado con `mlflow.sklearn.log_model` y nombre `WineQualityClassifier` |

---

## 🚀 Instalación y ejecución local

### 1. Clonar el repositorio

```bash
git clone https://github.com/TU_USUARIO/wine-ml-pipeline.git
cd wine-ml-pipeline
```

### 2. Instalar dependencias

```bash
pip install -r requirements.txt
```

O con make:

```bash
make install
```

### 3. Ejecutar el entrenamiento

```bash
python src/train.py
```

O con make:

```bash
make train
```

Al terminar se generan dos artefactos locales:

- `mlruns/` — experimentos y métricas registradas en MLflow
- `model.pkl` — modelo serializado listo para usar

### 4. Ejecutar pruebas de validación

```bash
python tests/test_model.py
```

O con make:

```bash
make test
```

### 5. Ejecutar todo junto

```bash
make all
```

### 6. Ver experimentos en MLflow UI

```bash
mlflow ui --backend-store-uri ./mlruns --port 5000
```

Abrir en el navegador:

http://localhost:5000


Desde ahí se pueden ver todos los runs, parámetros, métricas y el modelo registrado.

### 7. Limpiar artefactos generados

```bash
make clean
```

---

## 🤖 Pipeline de CI/CD con GitHub Actions

El archivo `.github/workflows/ml.yml` automatiza el pipeline completo en GitHub.
Se activa automáticamente en cada `push` o `pull_request` a la rama `main`,
y también puede ejecutarse de forma manual desde la pestaña **Actions**.

### Pasos del workflow

| Paso | Acción                      | Descripción                                          |
|------|-----------------------------|------------------------------------------------------|
| 1    | `actions/checkout@v4`       | Clona el repositorio en el runner de GitHub          |
| 2    | `actions/setup-python@v5`   | Configura Python 3.10 con caché de pip               |
| 3    | `make install`              | Instala todas las dependencias                       |
| 4    | `make train`                | Entrena y registra el modelo en MLflow               |
| 5    | `make test`                 | Valida métricas contra los umbrales mínimos          |
| 6    | `actions/upload-artifact@v4`| Sube `mlruns/` y `model.pkl` como artefacto del run  |

### Cómo ver el workflow corriendo

1. Haz push a la rama `main`
2. Ve a tu repositorio en GitHub
3. Entra a la pestaña **Actions**
4. Abre el workflow `ML Pipeline — Wine Quality CI/CD`
5. Verifica que todos los pasos estén en verde ✅
6. Al final, descarga el artefacto `wine-ml-artifacts-N` que contiene el modelo

---

## 📋 Comandos del Makefile

| Comando        | Descripción                                              |
|----------------|----------------------------------------------------------|
| `make install` | Instala dependencias desde `requirements.txt`            |
| `make train`   | Ejecuta el pipeline completo de entrenamiento            |
| `make test`    | Ejecuta pruebas de validación del modelo                 |
| `make lint`    | Verifica calidad del código con flake8                   |
| `make all`     | Ejecuta install → train → test en secuencia              |
| `make clean`   | Elimina `mlruns/` y `model.pkl` generados localmente     |

---

## ✅ Checklist del proyecto

- [x] Dataset externo, no de `sklearn.datasets`
- [x] Preprocesamiento: manejo de nulos, target binario, escalamiento, split
- [x] Modelo entrenado con LightGBM
- [x] Dos métricas evaluadas: Accuracy y F1-Score
- [x] MLflow registra parámetros, métricas, firma, input example y modelo
- [x] `src/train.py` funciona desde consola
- [x] `make install`, `make train`, `make test` funcionan
- [x] Pipeline CI/CD activo en GitHub Actions
- [x] Artefactos subidos al workflow
- [x] README con instrucciones claras

---

## 👤 Autor

**Oscar Mauricio Aragón**  
Maestría en Ciencia de Datos
Universidad EAN · 2026
