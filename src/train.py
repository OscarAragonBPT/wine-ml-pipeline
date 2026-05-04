"""
src/train.py - Pipeline completo de ML: carga, preprocesamiento,
entrenamiento con LightGBM, evaluacion y registro en MLflow.

Dataset : Wine Quality (Red) - UCI Machine Learning Repository
Modelo  : LGBMClassifier (clasificacion binaria: calidad buena/mala)
"""

import os
import sys
import traceback
import yaml
import joblib
import pandas as pd
import mlflow
import mlflow.sklearn
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.pipeline import Pipeline
from mlflow.models import infer_signature

# --- Cargar configuracion ---
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "config", "config.yaml")
with open(CONFIG_PATH, "r") as f:
    cfg = yaml.safe_load(f)

# --- Rutas ---
WORKSPACE = os.getcwd()
MLRUNS_DIR = os.path.join(WORKSPACE, "mlruns")
TRACKING_URI = "file://" + os.path.abspath(MLRUNS_DIR)
MODEL_PATH = os.path.join(WORKSPACE, "model.pkl")

print(f"Workspace : {WORKSPACE}")
print(f"MLRuns    : {MLRUNS_DIR}")
os.makedirs(MLRUNS_DIR, exist_ok=True)

# --- 1. Cargar datos ---
print("\nDescargando dataset Wine Quality (Red) desde UCI...")
df = pd.read_csv(cfg["data"]["url"], sep=cfg["data"]["separator"])
print(f"   Filas: {len(df)} | Columnas: {list(df.columns)}")

# --- 2. Preprocesamiento ---
print("\nPreprocesando datos...")
print(f"   Nulos antes: {df.isnull().sum().sum()}")
df.dropna(inplace=True)
print(f"   Nulos despues: {df.isnull().sum().sum()}")

# Variable objetivo binaria: calidad >= 7 es buena (1), < 7 es mala (0)
df["target"] = (df["quality"] >= 7).astype(int)
print(f"   Distribucion target: {df['target'].value_counts().to_dict()}")

X = df.drop(columns=["quality", "target"])
y = df["target"]

# Division train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=cfg["data"]["test_size"],
    random_state=cfg["data"]["random_state"],
    stratify=y
)
print(f"   Train: {X_train.shape} | Test: {X_test.shape}")

# --- 3. Pipeline sklearn ---
print("\nConstruyendo pipeline con StandardScaler + LGBMClassifier...")
model_params = {
    "n_estimators": cfg["model"]["n_estimators"],
    "max_depth": cfg["model"]["max_depth"],
    "learning_rate": cfg["model"]["learning_rate"],
    "num_leaves": cfg["model"]["num_leaves"],
    "random_state": cfg["model"]["random_state"],
    "verbose": -1,
}

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("lgbm", lgb.LGBMClassifier(**model_params)),
])

pipeline.fit(X_train, y_train)

# --- 4. Evaluacion ---
preds = pipeline.predict(X_test)
accuracy = accuracy_score(y_test, preds)
f1 = f1_score(y_test, preds)

print("\nResultados:")
print(f"   Accuracy : {accuracy:.4f}")
print(f"   F1-Score : {f1:.4f}")
print(f"\n{classification_report(y_test, preds, target_names=['Mala calidad', 'Buena calidad'])}")

# --- 5. Registro en MLflow ---
print("\nRegistrando en MLflow...")
mlflow.set_tracking_uri(TRACKING_URI)

exp_name = cfg["mlflow"]["experiment_name"]
try:
    experiment_id = mlflow.create_experiment(
        name=exp_name,
        artifact_location=TRACKING_URI,
    )
    print(f"   Experimento creado: {exp_name} (id={experiment_id})")
except mlflow.exceptions.MlflowException as exc:
    if "RESOURCE_ALREADY_EXISTS" in str(exc):
        exp = mlflow.get_experiment_by_name(exp_name)
        experiment_id = exp.experiment_id
        print(f"   Experimento existente: {exp_name} (id={experiment_id})")
    else:
        raise

input_example = X_test.iloc[:5]
signature = infer_signature(X_train, pipeline.predict(X_train))

try:
    with mlflow.start_run(experiment_id=experiment_id) as run:
        print(f"   Run ID: {run.info.run_id}")

        # Parametros
        mlflow.log_param("n_estimators", model_params["n_estimators"])
        mlflow.log_param("max_depth", model_params["max_depth"])
        mlflow.log_param("learning_rate", model_params["learning_rate"])
        mlflow.log_param("num_leaves", model_params["num_leaves"])
        mlflow.log_param("dataset", "Wine Quality Red UCI")
        mlflow.log_param("test_size", cfg["data"]["test_size"])

        # Metricas
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("f1_score", f1)

        # Modelo con firma e input_example
        mlflow.sklearn.log_model(
            sk_model=pipeline,
            artifact_path="model",
            signature=signature,
            input_example=input_example,
            registered_model_name=cfg["mlflow"]["model_name"],
        )

        print(f"   Modelo registrado - Accuracy={accuracy:.4f} | F1={f1:.4f}")

except Exception:
    traceback.print_exc()
    sys.exit(1)

# --- 6. Guardar model.pkl ---
joblib.dump(pipeline, MODEL_PATH)
print(f"\nmodel.pkl guardado en: {MODEL_PATH}")