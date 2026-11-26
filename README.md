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

Start the application:

    docker compose up --build

Swagger UI:

http://localhost:8000/docs

Health check:

http://localhost:8000/health

Stop the application:

    docker compose down

### Batch API Examples

CSV upload:

    curl.exe -X POST http://localhost:8000/api/v1/batch-audit -F "file=@data/sample_batch_24h.csv"

JSON input:

    curl.exe -X POST http://localhost:8000/api/v1/batch-audit -H "Content-Type: application/json" --data-binary "@data/sample24hr.json"

## 6. Reproduce the Pipeline

Run the following commands in order:

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

## 8. Project Structure

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

## 9. Limitations

- Models and engineering baselines are based on synthetic telemetry and require validation using real industrial data.
- Anomaly scores indicate unusual behaviour, not failure probabilities.
- Diagnostic recommendations indicate possible causes rather than confirmed equipment failures.
- A single reading cannot establish sensor freezing or long-term degradation.
- A 24-hour batch may not contain sufficient history for extended degradation analysis.
- Energy estimates are limited to detected periods with available operating baselines.

## Technology Stack

Python, Pandas, NumPy, Scikit-learn, Matplotlib, FastAPI, Pydantic, Joblib, Pytest and Docker.