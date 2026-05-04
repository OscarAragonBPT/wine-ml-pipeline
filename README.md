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
