import hashlib
import json
import joblib
import sklearn
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from src.data import ROOT, MODEL_PATH, FEATURES


def train():
    data = load_wine()
    indices = list(range(len(data.target)))
    train_idx, test_idx = train_test_split(indices, test_size=0.25,
                                          random_state=42, stratify=data.target)
    model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=1)
    model.fit(data.data[train_idx], data.target[train_idx])
    predicted = model.predict(data.data[test_idx])
    metrics = {"dataset": "sklearn.datasets.load_wine", "rows": len(indices),
               "features": FEATURES, "train_rows": len(train_idx), "test_rows": len(test_idx),
               "seed": 42, "model": "RandomForestClassifier", "n_estimators": 200,
               "accuracy": accuracy_score(data.target[test_idx], predicted),
               "macro_f1": f1_score(data.target[test_idx], predicted, average="macro"),
               "confusion_matrix": confusion_matrix(data.target[test_idx], predicted).tolist(),
               "sklearn_version": sklearn.__version__,
               "data_sha256": hashlib.sha256(data.data.tobytes()+data.target.tobytes()).hexdigest()}
    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump({"model": model, "features": FEATURES,
                 "classes": data.target_names.tolist(), "metrics": metrics}, MODEL_PATH)
    (ROOT / "evidence").mkdir(exist_ok=True)
    (ROOT / "evidence/metrics.json").write_text(json.dumps(metrics, indent=2))
    sample = dict(zip(FEATURES, data.data[test_idx[0]].tolist()))
    (ROOT / "evidence/sample.json").write_text(json.dumps(sample, indent=2))
    (ROOT / "evidence/heldout.json").write_text(json.dumps([
        {"input": dict(zip(FEATURES, data.data[i].tolist())), "expected": int(data.target[i])}
        for i in test_idx], indent=2))
    print(json.dumps(metrics, indent=2))
    print("Saved model:", MODEL_PATH)

if __name__ == "__main__":
    train()
