"""
Random Forest Classifier for flood risk prediction.
Trains on lat, lng, elevation and outputs a flood probability score (0.0 - 1.0).
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "flood_model.joblib")


class FloodMLModel:
    def __init__(self):
        self.model: RandomForestClassifier | None = None

    def train(self, df: pd.DataFrame, threshold: float) -> dict:
        """
        Trains a Random Forest model on the processed dataset.

        Features: lat, lng, elevation_y2
        Target: binary flood label (1 = Flooded, 0 = Safe)
        """
        required_cols = {"lat", "lng", "elevation_y2", "status_y2"}
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"Missing columns for ML training: {missing}")

        # Feature matrix & labels
        X = df[["lat", "lng", "elevation_y2"]].values
        # Using np.where to provide clear types for the linter
        y = np.where(df["status_y2"] == "Flooded", 1, 0)

        # Handle class imbalance gracefully
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        self.model = RandomForestClassifier(
            n_estimators=150,
            max_depth=12,
            min_samples_split=5,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )
        self.model.fit(X_train, y_train)

        # Evaluate
        y_pred = self.model.predict(X_test)
        accuracy = float(accuracy_score(y_test, y_pred))
        report = classification_report(y_test, y_pred, output_dict=True)

        # Persist model for fast reloads
        joblib.dump(self.model, MODEL_PATH)

        return {
            "accuracy": float(f"{accuracy * 100:.2f}"),
            "precision_flooded": float(f"{float(report.get('1', {}).get('precision', 0)) * 100:.2f}"),
            "recall_flooded": float(f"{float(report.get('1', {}).get('recall', 0)) * 100:.2f}"),
            "f1_flooded": float(f"{float(report.get('1', {}).get('f1-score', 0)) * 100:.2f}"),
            "training_samples": int(len(X_train)),
            "test_samples": int(len(X_test)),
            "model_status": "Trained & Saved",
        }

    def load_model(self) -> bool:
        """Load a previously saved model from disk."""
        if os.path.exists(MODEL_PATH):
            self.model = joblib.load(MODEL_PATH)
            return True
        return False

    def predict_probabilities(self, df: pd.DataFrame) -> list[dict]:
        """
        Returns flood probability scores for all rows.
        Each row gets a 'flood_probability' float (0.0 - 1.0).
        """
        if self.model is None:
            raise RuntimeError("Model is not trained or loaded. Call train() or load_model() first.")

        X = df[["lat", "lng", "elevation_y2"]].values
        probabilities = self.model.predict_proba(X)

        # predict_proba returns [[p_safe, p_flood], ...]
        flood_probs = probabilities[:, 1]

        # Merge back with coordinates
        result = []
        for i, row in df.iterrows():
            result.append({
                "lat": float(row["lat"]),
                "lng": float(row["lng"]),
                "flood_probability": float(f"{float(flood_probs[i if isinstance(i, int) else df.index.get_loc(i)]):.4f}"),
                "risk_level": _classify_risk(float(flood_probs[i if isinstance(i, int) else df.index.get_loc(i)])),
            })

        return result

    def predict_probabilities_fast(self, df: pd.DataFrame) -> pd.DataFrame:
        """Vectorized version — returns the processed dataframe with probability column appended."""
        if self.model is None:
            raise RuntimeError("Model not ready.")

        X = df[["lat", "lng", "elevation_y2"]].values
        probs = self.model.predict_proba(X)[:, 1]
        out = df.copy()
        out["flood_probability"] = probs
        out["risk_level"] = [_classify_risk(p) for p in probs]
        return out


def _classify_risk(prob: float) -> str:
    if prob < 0.25:
        return "Low"
    elif prob < 0.50:
        return "Moderate"
    elif prob < 0.75:
        return "High"
    else:
        return "Critical"
