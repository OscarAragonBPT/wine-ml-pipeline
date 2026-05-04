"""
tests/test_model.py - Pruebas basicas de validacion del modelo y datos.
Ejecutar con: python tests/test_model.py
"""

import os
import sys
import joblib
import pandas as pd
import yaml
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

print("\nIniciando pruebas de validacion...\n")
PASSED = 0
FAILED = 0


def check(name, condition, detail=""):
    global PASSED, FAILED
    if condition:
        print(f"  PASS - {name}")
        PASSED += 1
    else:
        print(f"  FAIL - {name} {detail}")
        FAILED += 1


# --- Cargar config ---
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "config", "config.yaml")
with open(CONFIG_PATH, "r") as f:
    cfg = yaml.safe_load(f)

# --- Test 1: Modelo existe ---
model_path = os.path.join(os.getcwd(), "model.pkl")
check("model.pkl existe", os.path.exists(model_path), f"(ruta: {model_path})")

if not os.path.exists(model_path):
    print("\nNo se puede continuar sin model.pkl. Ejecuta primero: make train")
    sys.exit(1)

model = joblib.load(model_path)
check("Modelo cargado correctamente", model is not None)

# --- Test 2: Dataset se puede descargar ---
try:
    df = pd.read_csv(cfg["data"]["url"], sep=cfg["data"]["separator"])
    check("Dataset descargado", len(df) > 0, f"({len(df)} filas)")
    check("Dataset tiene 12 columnas", len(df.columns) == 12, f"(columnas: {len(df.columns)})")
except Exception as e:
    check("Dataset descargado", False, str(e))
    sys.exit(1)

# --- Test 3: Preprocesamiento ---
df.dropna(inplace=True)
df["target"] = (df["quality"] >= 7).astype(int)
X = df.drop(columns=["quality", "target"])
y = df["target"]

check("Sin nulos tras limpieza", df.isnull().sum().sum() == 0)
check("Target es binario", set(y.unique()) == {0, 1})
check("X tiene 11 features", X.shape[1] == 11, f"(tiene {X.shape[1]})")

# --- Test 4: Predicciones y metricas ---
_, X_test, _, y_test = train_test_split(
    X, y,
    test_size=cfg["data"]["test_size"],
    random_state=cfg["data"]["random_state"],
    stratify=y
)

try:
    preds = model.predict(X_test)
    accuracy = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds)

    check("Modelo genera predicciones", len(preds) == len(y_test))
    check(
        f"Accuracy >= {cfg['thresholds']['accuracy']} (actual: {accuracy:.4f})",
        accuracy >= cfg["thresholds"]["accuracy"]
    )
    check(
        f"F1-Score >= {cfg['thresholds']['f1_score']} (actual: {f1:.4f})",
        f1 >= cfg["thresholds"]["f1_score"]
    )
except Exception as e:
    check("Predicciones sin error", False, str(e))

# --- Test 5: mlruns existe ---
mlruns_path = os.path.join(os.getcwd(), "mlruns")
check("Carpeta mlruns/ existe", os.path.isdir(mlruns_path))

# --- Resumen ---
total = PASSED + FAILED
print(f"\n{'='*45}")
print(f"  Resultado: {PASSED}/{total} pruebas pasadas")
print(f"{'='*45}")

if FAILED > 0:
    print(f"\n{FAILED} prueba(s) fallaron.")
    sys.exit(1)
else:
    print("\nTodas las pruebas pasaron. Pipeline listo.")
    sys.exit(0)