# Industrial Asset Energy & Maintenance Analytics

## Overview

An end-to-end industrial IoT analytics pipeline for detecting equipment anomalies, estimating energy wastage and generating maintenance recommendations.

The system monitors three industrial assets:

- Chiller-01 — Central Cooling Plant
- Compressor-02 — Compressed Air System
- Pump-03 — Cooling Water Circulation

## 1. Data Generation and Preprocessing

Generated 14 days of synthetic telemetry at one-minute intervals across three assets.

- Raw dataset: 60,450 records.
- Preprocessed dataset: 60,480 records.
- Simulated missing values, communication gaps, physical outliers, sensor freezing, power surges and gradual efficiency degradation.
- Applied conditional interpolation, physical boundary validation and sensor-freeze detection.
- Preserved data-quality flags and excluded unsuitable readings from ML evaluation.

**Engineered features:**

- `delta_t`
- `specific_energy`
- `rolling_power_mean_1h`
- `load_ratio`

## 2. Machine Learning

Isolation Forest and One-Class SVM were evaluated using standardized sensor measurements. Models were trained on eligible RUNNING observations to avoid false alarms during OFF and STANDBY periods.

An asset-specific model was selected based on validation performance.

| Asset | Selected Model | Test Precision | Test Recall | Test F1 |
|---|---|---:|---:|---:|
| Chiller-01 | One-Class SVM | 36.36% | 100.00% | 53.33% |
| Compressor-02 | One-Class SVM | 12.63% | 78.33% | 21.76% |
| Pump-03 | Isolation Forest | 97.95% | 83.10% | 89.92% |

One-Class SVM models nonlinear boundaries around normal operating conditions, while Isolation Forest identifies unusual observations through random partitioning.

Chronological data splitting was used:

- Training: October 1–8, 2026
- Validation: October 9–11, 2026
- Testing: October 12–14, 2026

Model selection and score calibration were completed before independent test evaluation.

### Anomaly Classification

Raw anomaly scores are normalized to a continuous range of 0–1 using training-derived calibration.

| Classification | Anomaly Score |
|---|---|
| NORMAL | Below 0.50 |
| WARNING | 0.50 to below 0.75 |
| CRITICAL | 0.75 and above |
| NOT_EVALUATED | Invalid, missing or ineligible telemetry |

The operational thresholds follow the assignment requirements. Threshold sensitivity analysis was performed to understand precision and recall trade-offs.

False alarms can lead to unnecessary maintenance, inspection costs and equipment downtime. Missed anomalies may result in equipment damage and energy loss. The Compressor model exhibits comparatively low precision, highlighting the need for further calibration with real operational data.

## 3. Operational Insights

Asset-specific engineering baselines are used to estimate expected power consumption.

Excess energy (kWh) = Sum of (Actual Power − Expected Power) × Interval Minutes / 60

Estimated cost loss = Excess Energy × $0.12/kWh

ML detection and diagnostic heuristics operate independently. A reading is flagged if either detects an anomaly; overlapping detections are counted only once. Consecutive flagged readings for the same asset are grouped into incidents.

Independent diagnostic heuristics identify:

- High power consumption without proportional flow increase.
- Temperature sensor freezing during RUNNING operation.
- Gradual efficiency degradation over extended operating periods.

### Sample 24-Hour Batch Results

| Metric | Result |
|---|---:|
| Input records | 4,300 |
| Preprocessed minute slots | 4,320 |
| ML anomalies | 112 |
| Additional rule-only anomalies | 29 |
| Total anomalous readings | 141 |
| Flagged incidents | 38 |
| Estimated excess energy | 12.01 kWh |
| Estimated cost loss | $1.44 |

The batch audit identified the injected 20-minute Chiller power surge and 60-minute Compressor temperature sensor freeze.

## 4. API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | API and model readiness |
| POST | `/api/v1/diagnose` | Single telemetry anomaly detection and diagnosis |
| POST | `/api/v1/batch-audit` | CSV or JSON batch analysis with energy and cost estimates |

Sample input files are available in the `data/` directory:

- `sample_batch_24h.csv`
- `sample24hr.json`

## 5. Quickstart

Prerequisites: Docker and Docker Compose.

Clone the repository and start the application (with Docker running):

```bash
git clone https://github.com/sowmi09/industrial-asset-analytics.git
cd industrial-asset-analytics
docker compose up --build
```

The API uses the saved model artifacts in `models/`; retraining is not required when these artifacts are present.

Swagger UI:

http://localhost:8000/docs

Health check:

http://localhost:8000/health

Stop the application:

    docker compose down

## 6. Reproduce the Pipeline

Optional: regenerate the data, models and analysis outputs from scratch. Stop the API first with `docker compose down` if it is running. Run the following commands in order:

    docker compose run --rm analytics python src/generate_data.py
    docker compose run --rm analytics python src/data_processor.py
    docker compose run --rm analytics python src/visualize.py
    docker compose run --rm analytics python src/analyze_features.py
    docker compose run --rm analytics python src/train_eval.py
    docker compose run --rm analytics python src/train_eval.py --score-test
    docker compose run --rm analytics python src/insights.py

## 7. Automated Testing

    docker compose run --rm analytics python -m pytest tests/ -v

Tests cover missing timestamp restoration, physical outlier handling, normal API prediction and critical anomaly diagnosis.

## 8. API Testing — Sample Inputs and Outputs

With the API running, open http://localhost:8000/docs. Expand `POST /api/v1/diagnose`, click **Try it out**, paste a JSON input and click **Execute**.

### Test 1 — Health Check

Open http://localhost:8000/health.

```json
{
  "status": "healthy",
  "models_loaded": ["Chiller-01", "Compressor-02", "Pump-03"]
}
```

### Test 2 — Normal Operation

Input:

```json
{
  "timestamp": "2026-10-01T14:30:00Z",
  "asset_id": "Chiller-01",
  "power_kw": 97.39,
  "flow_rate": 400.0,
  "supply_temp_c": 6.7,
  "return_temp_c": 11.7,
  "pressure_bar": 2.50,
  "status": "RUNNING"
}
```

Observed response:

```json
{
  "anomaly_score": 0.45695393427333764,
  "classification": "NORMAL",
  "diagnostic_message": "No abnormal pattern detected in the current reading. Continue routine monitoring. A single reading cannot establish sensor freezing or long-term degradation."
}
```

### Test 3 — Warning

Reuse the complete input from Test 2, changing only `power_kw` to `110.0`.

Observed classification: `WARNING`. Expected score range: 0.50 to below 0.75. The exact score was not recorded.

### Test 4 — Critical High-Power Anomaly

Reuse the complete input from Test 2, changing only `power_kw` to `150.0`.

Observed response:

```json
{
  "anomaly_score": 0.9444558289507292,
  "classification": "CRITICAL",
  "diagnostic_message": "High power consumption detected without proportional flow/operating-load increase. Inspect motor load and condenser conditions. A single reading cannot establish sensor freezing or long-term degradation."
}
```

### Test 5 — Critical Anomaly Outside Supported Baseline Conditions

Input (assignment example):

```json
{
  "timestamp": "2026-10-01T14:30:00Z",
  "asset_id": "Chiller-01",
  "power_kw": 285.5,
  "flow_rate": 450.0,
  "supply_temp_c": 7.2,
  "return_temp_c": 12.8,
  "pressure_bar": 4.2,
  "status": "RUNNING"
}
```

Observed response:

```json
{
  "anomaly_score": 1,
  "classification": "CRITICAL",
  "diagnostic_message": "ML detected unusual telemetry; no specific diagnostic heuristic confirmed. Review sensor quality and operating conditions before maintenance. Expected-power baseline unavailable for these operating conditions. A single reading cannot establish sensor freezing or long-term degradation."
}
```

ML can flag an anomaly even when a specific engineering baseline or diagnosis is unavailable.

### Test 6 — Invalid Sensor Reading

Reuse the complete input from Test 2, changing only `power_kw` to `-10.0`.

Observed response:

```json
{
  "anomaly_score": null,
  "classification": "NOT_EVALUATED",
  "diagnostic_message": "Reading not evaluated: equipment is not RUNNING or sensor inputs are missing or invalid. A single reading cannot establish sensor freezing or long-term degradation."
}
```

Invalid telemetry is not assigned a misleading ML anomaly score.

### Test 7 — Batch CSV Audit

Submit `data/sample_batch_24h.csv` to `POST /api/v1/batch-audit` through Swagger UI, or try the following command from the repository directory:

```bash
curl -X POST http://localhost:8000/api/v1/batch-audit -F "file=@data/sample_batch_24h.csv"
```

Observed response summary (excluding `flagged_intervals` and the explanatory note):

```json
{
  "total_records_analyzed": 4300,
  "minute_slots_after_preprocessing": 4320,
  "scored_records": 3203,
  "unscored_records": 1117,
  "anomaly_count": 141,
  "ml_anomaly_count": 112,
  "rule_only_count": 29,
  "incident_count": 38,
  "estimated_excess_energy_kwh": 12.011435437065572,
  "estimated_cost_loss_usd": 1.4413722524478687,
  "energy_evaluated_minutes": 141,
  "energy_unavailable_minutes": 0
}
```

Key incidents (October 15, 2026, UTC):

| Asset | Flagged interval | Finding | Duration | Excess energy |
|---|---|---|---:|---:|
| Chiller-01 | 12:00–12:19 | High power consumption | 20 min | 12.01 kWh |
| Compressor-02 | 14:00–14:59 | Return temperature sensor freeze | 60 min | 0 kWh |

The full response includes incident timestamps, diagnostic messages and maintenance recommendations.

### Test 8 — Batch JSON Audit

Submit the equivalent JSON sample:

```bash
curl -X POST http://localhost:8000/api/v1/batch-audit -H "Content-Type: application/json" --data-binary "@data/sample24hr.json"
```

For Windows PowerShell, use `curl.exe` instead of `curl`. Alternatively, the following PowerShell command was used successfully:

```powershell
Invoke-RestMethod -Uri "http://localhost:8000/api/v1/batch-audit" -Method Post -ContentType "application/json" -InFile "data/sample24hr.json"
```

Both input formats were manually tested successfully. The JSON sample contains the same telemetry, with missing sensor values represented as `null`. The exact JSON response totals were not recorded; compare them against Test 7 to verify numerical consistency. The curl examples are provided as alternatives to the manually tested methods.

The recorded scores above are reference results using the saved models; small numerical differences may occur across environments or after retraining.

## 9. Project Structure

| File / Directory | Purpose |
|---|---|
| `data/` | Synthetic telemetry and sample API inputs |
| `models/` | Serialized models, scalers and calibration metadata |
| `outputs/` | Visualizations, evaluation results and operational insights |
| `src/generate_data.py` | Synthetic data generation |
| `src/data_processor.py` | Cleaning and feature engineering |
| `src/visualize.py` | Exploratory visualizations |
| `src/analyze_features.py` | Feature correlation analysis |
| `src/train_eval.py` | Model training, selection and evaluation |
| `src/insights.py` | Energy calculations and diagnostic heuristics |
| `src/main.py` | FastAPI service |
| `tests/` | Automated tests |
| `requirements.txt` | Pinned Python dependencies |
| `Dockerfile` | Container environment |
| `docker-compose.yaml` | Application deployment |

## 10. Limitations

- Models and engineering baselines are based on synthetic telemetry and require validation using real industrial data.
- Anomaly scores indicate unusual behaviour, not failure probabilities.
- Diagnostic recommendations indicate possible causes rather than confirmed equipment failures.
- A single reading cannot establish sensor freezing or long-term degradation.
- A 24-hour batch may not contain sufficient history for extended degradation analysis.
- Energy estimates are limited to detected periods with available operating baselines.

## Technology Stack

Python, Pandas, NumPy, Scikit-learn, Matplotlib, FastAPI, Pydantic, Joblib, Pytest and Docker.