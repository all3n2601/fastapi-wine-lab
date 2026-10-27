import json
import pytest
from fastapi.testclient import TestClient
from src.main import app
from src.data import ROOT
from src import store

@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "DB_PATH", tmp_path / "history.sqlite3")
    with TestClient(app) as c:
        yield c

@pytest.fixture
def sample():
    return json.loads((ROOT / "evidence/sample.json").read_text())

def test_health(client):
    assert client.get("/").json()["status"] == "healthy"

def test_prediction(client, sample):
    result = client.post("/predict", json=sample)
    assert result.status_code == 200
    body = result.json()
    assert body["predicted_class"] in [0, 1, 2]
    assert sum(body["probabilities"].values()) == pytest.approx(1)
    assert body["cultivar"] == f'class_{body["predicted_class"]}'

@pytest.mark.parametrize("change", ["missing", "negative", "text", "extra"])
def test_invalid(client, sample, change):
    if change == "missing": sample.pop("alcohol")
    if change == "negative": sample["alcohol"] = -1
    if change == "text": sample["alcohol"] = "invalid"
    if change == "extra": sample["unknown"] = 1
    assert client.post("/predict", json=sample).status_code == 422

def test_order_independent(client, sample):
    first = client.post("/predict", json=sample).json()
    second = client.post("/predict", json=dict(reversed(list(sample.items())))).json()
    first.pop("prediction_id")
    second.pop("prediction_id")
    assert first == second

def test_docs(client):
    assert client.get("/docs").status_code == 200
    assert "/predict" in client.get("/openapi.json").json()["paths"]

def test_model_metadata(client):
    assert client.get("/model-info").json()["test_rows"] == 45


def test_history_crud(client, sample):
    identifier = client.post("/predict", json=sample).json()["prediction_id"]
    path = f"/predictions/{identifier}"
    assert client.get(path).json()["input"] == sample
    assert client.patch(path, json={"note": "Reviewed sample"}).json()["note"] == "Reviewed sample"
    assert client.get("/predictions?limit=1&offset=0").json()["total"] == 1
    assert client.delete(path).status_code == 204
    assert client.get(path).status_code == 404
    assert client.delete(path).status_code == 404

def test_batch(client, sample):
    result = client.post("/predict/batch", json={"samples": [sample, sample]})
    assert result.status_code == 200
    assert result.json()["count"] == 2
    assert len({x["prediction_id"] for x in result.json()["predictions"]}) == 2
    assert client.post("/predict/batch", json={"samples": []}).status_code == 422
    assert client.post("/predict/batch", json={"samples": [sample]*101}).status_code == 422

def test_batch_validation_has_no_partial_writes(client, sample):
    result = client.post("/predict/batch", json={"samples": [sample, {"alcohol": -1}]})
    assert result.status_code == 422
    assert client.get("/predictions").json()["total"] == 0

def test_csv(client, sample):
    import io, csv
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=list(sample))
    writer.writeheader()
    writer.writerow(sample)
    result = client.post("/predict/csv", files={"file": ("sample.csv", stream.getvalue(), "text/csv")})
    assert result.status_code == 200
    assert result.json()["count"] == 1
    assert client.post("/predict/csv", files={"file": ("bad.csv", "wrong\n1", "text/csv")}).status_code == 422

def test_csv_size_limit(client):
    assert client.post("/predict/csv", files={"file": ("big.csv", b"a"*1_000_001)}).status_code == 413

def test_background_job(client, sample):
    result = client.post("/jobs", json={"samples": [sample, sample]})
    assert result.status_code == 202
    job = client.get(result.json()["status_url"]).json()
    assert job["status"] == "completed"
    assert len(job["result"]) == 2
    assert client.get("/jobs/unknown").status_code == 404

def test_middleware_errors_metrics(client):
    response = client.get("/predictions/9999")
    assert response.status_code == 404
    assert response.headers["x-request-id"] == response.json()["request_id"]
    assert float(response.headers["x-process-time-ms"]) >= 0
    assert client.get("/metrics").json()["errors_this_process"] >= 1

def test_history_query_validation(client):
    for query in ["limit=0", "offset=-1", "cultivar=unknown"]:
        assert client.get("/predictions?"+query).status_code == 422

def test_job_restart_recovery(client):
    with store.connect() as db:
        db.execute("INSERT INTO jobs(id,status) VALUES ('interrupted','running')")
    store.initialize()
    assert client.get("/jobs/interrupted").json()["status"] == "failed"

def test_persistence_across_app_restart(client, sample):
    identifier = client.post("/predict", json=sample).json()["prediction_id"]
    with TestClient(app) as restarted:
        assert restarted.get(f"/predictions/{identifier}").status_code == 200

def test_history_filter_pagination(client, sample):
    for _ in range(3): client.post('/predict', json=sample)
    page = client.get('/predictions?cultivar=class_0&limit=1&offset=1').json()
    assert page['total'] == 3
    assert len(page['items']) == 1
    assert page['items'][0]['id'] == 2
    assert client.get('/predictions?cultivar=class_2').json()['total'] == 0

def test_csv_bad_values_no_partial_writes(client, sample):
    import io, csv
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=list(sample))
    writer.writeheader()
    writer.writerow(sample)
    writer.writerow({**sample, 'alcohol': -1})
    assert client.post('/predict/csv', files={'file': ('bad.csv', stream.getvalue())}).status_code == 422
    assert client.get('/predictions').json()['total'] == 0

def test_background_failure(client, sample, monkeypatch):
    import importlib
    module = importlib.import_module('src.main')
    def fail(*args): raise RuntimeError('test failure')
    monkeypatch.setattr(module, 'infer', fail)
    result = client.post('/jobs', json={'samples': [sample]})
    assert client.get(result.json()['status_url']).json()['status'] == 'failed'
