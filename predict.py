"""
Inference utilities: single-patient prediction + attention visualization.
"""

import torch
import numpy as np
from model import HeartDiseaseTransformer
from data import FEATURE_NAMES, FEATURE_DESCRIPTIONS


def load_model(path: str = "heart_transformer.pth", device: str = "cpu"):
    checkpoint = torch.load(path, map_location=device)
    model = HeartDiseaseTransformer(**checkpoint["config"])
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model


def predict(model, scaler, patient_features: dict, device: str = "cpu"):
    """
    Predict heart disease risk for a single patient.

    Parameters
    ----------
    patient_features : dict mapping feature name → value
    Returns: (probability, prediction_label, attention_weights)
    """
    x_raw = np.array([patient_features[f] for f in FEATURE_NAMES], dtype=np.float32)
    x_scaled = scaler.transform(x_raw.reshape(1, -1))
    x_tensor = torch.FloatTensor(x_scaled).to(device)

    with torch.no_grad():
        logits = model(x_tensor)
        probs  = torch.softmax(logits, dim=1).squeeze().cpu().numpy()
        pred   = int(np.argmax(probs))
        attn   = model.get_attention_weights(x_tensor).squeeze().cpu().numpy()

    # CLS → feature attention (row 0, cols 1:)
    cls_attn = attn[0, 1:]  # shape: (num_features,)
    cls_attn_norm = cls_attn / (cls_attn.sum() + 1e-8)

    return {
        "probability_disease": float(probs[1]),
        "probability_healthy": float(probs[0]),
        "prediction": "Disease" if pred == 1 else "Healthy",
        "risk_level": "High" if probs[1] > 0.7 else "Medium" if probs[1] > 0.4 else "Low",
        "feature_importance": {
            FEATURE_NAMES[i]: float(cls_attn_norm[i])
            for i in range(len(FEATURE_NAMES))
        },
        "top_features": sorted(
            FEATURE_NAMES,
            key=lambda f: cls_attn_norm[FEATURE_NAMES.index(f)],
            reverse=True
        )[:5],
    }


# Example patient profiles for testing
SAMPLE_PATIENTS = {
    "high_risk": {
        "age": 65, "sex": 1, "cp": 3, "trestbps": 160, "chol": 310,
        "fbs": 1, "restecg": 2, "thalach": 100, "exang": 1,
        "oldpeak": 4.0, "slope": 2, "ca": 3, "thal": 3
    },
    "low_risk": {
        "age": 35, "sex": 0, "cp": 0, "trestbps": 115, "chol": 180,
        "fbs": 0, "restecg": 0, "thalach": 175, "exang": 0,
        "oldpeak": 0.1, "slope": 0, "ca": 0, "thal": 2
    },
    "moderate_risk": {
        "age": 52, "sex": 1, "cp": 1, "trestbps": 140, "chol": 260,
        "fbs": 0, "restecg": 1, "thalach": 140, "exang": 0,
        "oldpeak": 1.8, "slope": 1, "ca": 1, "thal": 2
    }
}


if __name__ == "__main__":
    from train import train
    from sklearn.preprocessing import StandardScaler

    print("Training model for inference demo...")
    model, _, scaler = train(epochs=40)

    for profile_name, features in SAMPLE_PATIENTS.items():
        result = predict(model, scaler, features)
        print(f"\n[{profile_name.upper()}]")
        print(f"  Prediction     : {result['prediction']} ({result['risk_level']} Risk)")
        print(f"  Disease Prob   : {result['probability_disease']:.3f}")
        print(f"  Top features   : {result['top_features']}")
