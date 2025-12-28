import pandas as pd
from joblib import load
from pathlib import Path

# Base directory = project root
BASE_DIR = Path(__file__).resolve().parents[2]
# Explanation:
# utils/load_data.py -> parents[0]=utils
# parents[1]=app
# parents[2]=Customer_Segmentation (root)

DATASET_DIR = BASE_DIR / "dataset"
MODELS_DIR = BASE_DIR / "models"


def load_data():
    df = pd.read_csv(DATASET_DIR / "Clustered_df.csv")
    pca = pd.read_csv(DATASET_DIR / "pca_2d.csv")
    return df, pca


def load_models():
    with open(MODELS_DIR / "scaler.pkl", "rb") as f:
        scaler = load(f)

    with open(MODELS_DIR / "kmeans.pkl", "rb") as f:
        kmeans = load(f)

    return scaler, kmeans
