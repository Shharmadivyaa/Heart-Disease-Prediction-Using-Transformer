"""
Data preprocessing for Heart Disease (Cleveland UCI) dataset.

Features (13):
  1. age          - Age in years
  2. sex          - (1=male, 0=female)
  3. cp           - Chest pain type (0-3)
  4. trestbps     - Resting blood pressure (mm Hg)
  5. chol         - Serum cholesterol (mg/dl)
  6. fbs          - Fasting blood sugar > 120 mg/dl (1=true)
  7. restecg      - Resting ECG results (0-2)
  8. thalach      - Maximum heart rate achieved
  9. exang        - Exercise induced angina (1=yes)
  10. oldpeak      - ST depression induced by exercise
  11. slope        - Slope of peak exercise ST segment (0-2)
  12. ca           - Number of major vessels (0-3)
  13. thal         - Thalassemia (0-3)

Target: 0 = no disease, 1 = disease present
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import Dataset, DataLoader


FEATURE_NAMES = [
    "age", "sex", "cp", "trestbps", "chol", "fbs",
    "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"
]

FEATURE_DESCRIPTIONS = {
    "age":      "Age (years)",
    "sex":      "Sex (1=Male, 0=Female)",
    "cp":       "Chest Pain Type (0-3)",
    "trestbps": "Resting Blood Pressure (mm Hg)",
    "chol":     "Cholesterol (mg/dL)",
    "fbs":      "Fasting Blood Sugar > 120 (1=True)",
    "restecg":  "Resting ECG (0-2)",
    "thalach":  "Max Heart Rate",
    "exang":    "Exercise Angina (1=Yes)",
    "oldpeak":  "ST Depression",
    "slope":    "ST Slope (0-2)",
    "ca":       "Major Vessels (0-3)",
    "thal":     "Thalassemia (0-3)",
}

# Realistic mean values for synthetic data generation (Cleveland dataset statistics)
FEATURE_STATS = {
    "age":      (54.4, 9.0,   29, 77),
    "sex":      (0.68, 0.47,  0,  1),
    "cp":       (0.97, 1.03,  0,  3),
    "trestbps": (131.6, 17.5, 94, 200),
    "chol":     (246.3, 51.8, 126, 564),
    "fbs":      (0.15, 0.36,  0,  1),
    "restecg":  (0.53, 0.53,  0,  2),
    "thalach":  (149.6, 22.9, 71, 202),
    "exang":    (0.33, 0.47,  0,  1),
    "oldpeak":  (1.04, 1.16,  0,  6.2),
    "slope":    (1.40, 0.62,  0,  2),
    "ca":       (0.73, 1.02,  0,  3),
    "thal":     (2.31, 0.61,  0,  3),
}


def generate_synthetic_dataset(n_samples: int = 1000, random_state: int = 42) -> pd.DataFrame:
    """
    Generate a realistic synthetic dataset mimicking Cleveland Heart Disease data.
    Uses clinically informed correlations between features and disease.
    """
    rng = np.random.RandomState(random_state)
    data = {}

    for feat, (mean, std, low, high) in FEATURE_STATS.items():
        if feat in ("sex", "fbs", "exang"):
            data[feat] = rng.binomial(1, mean, n_samples).astype(float)
        elif feat in ("cp", "restecg", "slope"):
            data[feat] = rng.randint(int(low), int(high) + 1, n_samples).astype(float)
        elif feat == "ca":
            data[feat] = rng.randint(0, 4, n_samples).astype(float)
        elif feat == "thal":
            data[feat] = rng.choice([0, 1, 2, 3], n_samples, p=[0.05, 0.05, 0.72, 0.18]).astype(float)
        else:
            vals = rng.normal(mean, std, n_samples)
            vals = np.clip(vals, low, high)
            data[feat] = vals

    df = pd.DataFrame(data)

    # Clinically informed target: disease risk score
    risk = (
        0.03 * (df["age"] - 54) +
        0.4  * df["sex"] +
        0.5  * (df["cp"] > 0).astype(float) +
        0.003 * (df["trestbps"] - 130) +
        0.001 * (df["chol"] - 200) +
        0.3  * df["fbs"] -
        0.02 * (df["thalach"] - 150) +
        0.5  * df["exang"] +
        0.4  * df["oldpeak"] +
        0.3  * df["ca"] +
        0.4  * (df["thal"] == 3).astype(float) +
        rng.normal(0, 0.5, n_samples)
    )

    threshold = np.percentile(risk, 54)  # ~46% positive class
    df["target"] = (risk > threshold).astype(int)

    return df


class HeartDiseaseDataset(Dataset):
    """PyTorch Dataset for heart disease data."""

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def prepare_data(n_samples: int = 1200, test_size: float = 0.2, val_size: float = 0.1,
                 batch_size: int = 32, random_state: int = 42):
    """Full pipeline: generate → split → scale → DataLoaders."""
    df = generate_synthetic_dataset(n_samples, random_state)

    X = df[FEATURE_NAMES].values.astype(np.float32)
    y = df["target"].values.astype(np.int64)

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    val_ratio = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_ratio,
        random_state=random_state, stratify=y_train_val
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val   = scaler.transform(X_val)
    X_test  = scaler.transform(X_test)

    train_loader = DataLoader(HeartDiseaseDataset(X_train, y_train),
                              batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(HeartDiseaseDataset(X_val, y_val),
                              batch_size=batch_size)
    test_loader  = DataLoader(HeartDiseaseDataset(X_test, y_test),
                              batch_size=batch_size)

    return train_loader, val_loader, test_loader, scaler, df


if __name__ == "__main__":
    df = generate_synthetic_dataset()
    print(df.describe())
    print(f"\nClass distribution:\n{df['target'].value_counts()}")
