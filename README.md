
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
Oscar Mauricio Aragón Morales

**Oscar Mauricio Aragón**  
Maestría en Inteligencia Artificial — MLOps  
Universidad EAN · 2026
