from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "sample_feedback.csv"
LABELS = ["Positive Feedback", "Negative Feedback", "Question", "Suggestion", "Complaint"]


def load_data(path: Path = DATA_PATH) -> tuple[pd.Series, pd.Series]:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    data = pd.read_csv(path)
    if not {"text", "label"}.issubset(data.columns):
        raise ValueError("Dataset must contain 'text' and 'label' columns.")
    data = data.dropna(subset=["text", "label"])
    data["text"] = data["text"].astype(str).str.strip()
    data = data[data["text"] != ""]
    if data.empty:
        raise ValueError("Dataset contains no usable feedback examples.")
    unknown = sorted(set(data["label"]) - set(LABELS))
    if unknown:
        raise ValueError(f"Unknown label(s) in dataset: {unknown}")
    counts = data["label"].value_counts()
    if len(counts) < 2 or counts.min() < 2:
        raise ValueError("Need at least two examples per category to train/evaluate.")
    return data["text"], data["label"]


def build_model() -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), lowercase=True)),
        ("classifier", LogisticRegression(max_iter=2000, random_state=42)),
    ])


def train_and_evaluate() -> tuple[Pipeline, dict[str, float]]:
    texts, labels = load_data()
    x_train, x_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.25, random_state=42, stratify=labels
    )
    model = build_model()
    model.fit(x_train, y_train)
    predicted = model.predict(x_test)
    metrics = {
        "accuracy": float(accuracy_score(y_test, predicted)),
        "macro_f1": float(f1_score(y_test, predicted, average="macro")),
    }
    print("Held-out evaluation")
    print(f"Accuracy: {metrics['accuracy']:.3f}")
    print(f"Macro-F1: {metrics['macro_f1']:.3f}")
    print(classification_report(y_test, predicted, zero_division=0))
    return model, metrics


def classify(model: Pipeline, text: str, threshold: float = 0.45) -> dict:
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Please enter a non-empty feedback message.")
    if len(text) > 2000:
        raise ValueError("Feedback is too long. Limit input to 2,000 characters.")
    if not 0 <= threshold <= 1:
        raise ValueError("Confidence threshold must be between 0 and 1.")
    probabilities = model.predict_proba([text.strip()])[0]
    index = int(probabilities.argmax())
    confidence = float(probabilities[index])
    label = str(model.classes_[index])
    return {
        "input": text.strip(),
        "category": label,
        "confidence": round(confidence, 4),
        "needs_human_review": confidence < threshold,
        "review_reason": f"Confidence is below the {threshold:.2f} review threshold." if confidence < threshold else None,
    }
