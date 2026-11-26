import pandas as pd
from fastapi.testclient import TestClient

from src.main import app
from src.data_processor import process_telemetry


NORMAL_READING = {
    "timestamp": "2026-10-01T14:30:00Z",
    "asset_id": "Chiller-01",
    "power_kw": 97.39,
    "flow_rate": 400.0,
    "supply_temp_c": 6.7,
    "return_temp_c": 11.7,
    "pressure_bar": 2.50,
    "status": "RUNNING"
}


# Test 1: Missing timestamp restoration
def test_missing_timestamp():
    first = NORMAL_READING.copy()

    last = {
        **NORMAL_READING,
        "timestamp": "2026-10-01T14:32:00Z"
    }

    df = pd.DataFrame([first, last])
    result = process_telemetry(df)

    assert len(result) == 3
    assert result["is_missing_timestamp"].sum() == 1


# Test 2: Negative power validation
def test_negative_power():
    reading = {
        **NORMAL_READING,
        "power_kw": -10.0
    }

    df = pd.DataFrame([reading])
    result = process_telemetry(df)

    assert pd.isna(result.iloc[0]["power_kw"])
    assert not result.iloc[0]["is_ml_eligible"]


# Test 3: Normal API prediction
def test_normal_diagnosis():
    with TestClient(app) as client:
        response = client.post(
            "/api/v1/diagnose",
            json=NORMAL_READING
        )

    assert response.status_code == 200

    result = response.json()

    assert result["classification"] == "NORMAL"
    assert 0 <= result["anomaly_score"] < 0.50


# Test 4: Critical API prediction
def test_critical_diagnosis():
    reading = {
        **NORMAL_READING,
        "power_kw": 150.0
    }

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/diagnose",
            json=reading
        )

    assert response.status_code == 200

    result = response.json()

    assert result["classification"] == "CRITICAL"
    assert 0.75 <= result["anomaly_score"] <= 1.0
    assert "high power" in result["diagnostic_message"].lower()