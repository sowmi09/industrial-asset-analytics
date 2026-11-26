"""FastAPI inference service; run from repository root with uvicorn src.main:app."""
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal
import io
import logging
import os
import math
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, ConfigDict, ValidationError, TypeAdapter

if __package__:
    from .data_processor import process_telemetry
    from .insights import load_bundles, audit, diagnosis
else:
    from data_processor import process_telemetry
    from insights import load_bundles, audit, diagnosis

MAX_ROWS = 4320
MAX_BYTES = 5 * 1024 * 1024
ROOT = Path(__file__).resolve().parent.parent

class Telemetry(BaseModel):
    model_config = ConfigDict(extra='forbid', allow_inf_nan=False)
    timestamp: str
    asset_id: Literal['Chiller-01', 'Compressor-02', 'Pump-03']
    power_kw: float | None
    flow_rate: float | None
    supply_temp_c: float | None
    return_temp_c: float | None
    pressure_bar: float | None
    status: Literal['RUNNING', 'STANDBY', 'OFF']

class DiagnosisResponse(BaseModel):
    anomaly_score: float | None
    classification: Literal['NORMAL', 'WARNING', 'CRITICAL', 'NOT_EVALUATED']
    diagnostic_message: str

@asynccontextmanager
async def lifespan(app):
    app.state.bundles = {}
    try:
        app.state.bundles = load_bundles(os.getenv('MODEL_DIR', str(ROOT / 'models')))
    except Exception:
        logging.exception('Model loading failed; service is not ready')
    yield
    app.state.bundles = {}

app = FastAPI(title='Industrial Asset Analytics', version='1.0.0', lifespan=lifespan)

def bundles(request):
    loaded = getattr(request.app.state, 'bundles', {})
    if len(loaded) != 3:
        raise HTTPException(503, 'Model bundles unavailable or incompatible')
    return loaded

@app.get('/health')
def health(request: Request):
    from fastapi.responses import JSONResponse
    loaded = getattr(request.app.state, 'bundles', {})
    ready = len(loaded) == 3
    return JSONResponse({'status': 'healthy' if ready else 'not_ready',
                         'models_loaded': sorted(loaded)}, status_code=200 if ready else 503)

def run_audit(rows, loaded, retrospective):
    try:
        raw = pd.DataFrame(rows)
        ts = pd.to_datetime(raw.timestamp, utc=True, errors='raise')
        if ts.max() - ts.min() >= pd.Timedelta(hours=24):
            raise ValueError('Batch must span less than 24 hours (1440 minute slots)')
        cleaned = process_telemetry(raw, retrospective=retrospective)
        return audit(cleaned, loaded, retrospective=retrospective)
    except (ValueError, TypeError) as exc:
        raise HTTPException(422, str(exc)) from exc

@app.post('/api/v1/diagnose', response_model=DiagnosisResponse)
def diagnose(reading: Telemetry, request: Request):
    g, _ = run_audit([reading.model_dump()], bundles(request), False)
    row = g.iloc[0]
    if row.rule_detected or row.ml_detected:
        message, action = diagnosis(g)
        message += ' ' + action
    elif row.classification == 'NOT_EVALUATED':
        message = 'Reading not evaluated: equipment is not RUNNING or sensor inputs are missing or invalid.'
    else:
        message = 'No abnormal pattern detected in the current reading. Continue routine monitoring.'
    if pd.isna(row.expected_power_kw) and reading.status == 'RUNNING':
        message += ' Expected-power baseline unavailable for these operating conditions.'
    message += ' A single reading cannot establish sensor freezing or long-term degradation.'
    return DiagnosisResponse(anomaly_score=None if pd.isna(row.anomaly_score) else float(row.anomaly_score),
                             classification=row.classification, diagnostic_message=message)

def finite(value):
    return float(value) if pd.notna(value) and math.isfinite(float(value)) else None

def batch_result(rows, loaded):
    g, periods = run_audit(rows, loaded, True)
    detected = g[g.detected]
    net = detected.excess_energy_kwh.sum(min_count=1) if len(detected) else 0.0
    cost = periods.estimated_cost_loss_usd.sum(min_count=1) if len(periods) else 0.0
    intervals = []
    for _, p in periods.iterrows():
        intervals.append({'asset_id': p.asset_id, 'start_utc': p.start_utc.isoformat(),
                          'end_utc': p.end_utc.isoformat(), 'duration_minutes': int(p.duration_minutes),
                          'ml_severity': p.ml_severity, 'diagnostic_message': p.diagnostic_message,
                          'recommended_action': p.recommended_action,
                          'excess_energy_kwh': finite(p.net_excess_energy_kwh),
                          'estimated_cost_loss_usd': finite(p.estimated_cost_loss_usd)})
    return {'total_records_analyzed': len(rows), 'minute_slots_after_preprocessing': len(g),
            'scored_records': int(g.anomaly_score.notna().sum()),
            'unscored_records': int(g.anomaly_score.isna().sum()),
            'anomaly_count': int(g.detected.sum()), 'ml_anomaly_count': int(g.ml_detected.sum()),
            'rule_only_count': int((g.rule_detected & ~g.ml_detected).sum()),
            'incident_count': len(intervals), 'estimated_excess_energy_kwh': finite(net),
            'estimated_cost_loss_usd': finite(cost),
            'energy_evaluated_minutes': int(detected.excess_energy_kwh.notna().sum()),
            'energy_unavailable_minutes': int(detected.excess_energy_kwh.isna().sum()),
            'flagged_intervals': intervals,
            'note': 'Retrospective audit. A 24-hour batch alone may lack sufficient history to establish a 24+ hour degradation trend. Energy totals cover evaluable detected minutes only.'}

@app.post('/api/v1/batch-audit', openapi_extra={'requestBody': {'required': True, 'content': {
    'application/json': {'schema': {'type': 'array', 'items': Telemetry.model_json_schema()}},
    'multipart/form-data': {'schema': {'type': 'object', 'required': ['file'], 'properties': {'file': {'type': 'string', 'format': 'binary'}}}}
}}})
async def batch_audit(request: Request):
    loaded = bundles(request)
    content_type = request.headers.get('content-type', '').lower()
    try:
        if int(request.headers.get('content-length', '0')) > MAX_BYTES:
            raise HTTPException(413, 'Payload too large')
        if content_type.startswith('application/json'):
            body = await request.body()
            if len(body) > MAX_BYTES:
                raise HTTPException(413, 'Payload too large')
            rows = TypeAdapter(list[Telemetry]).validate_json(body)
        elif content_type.startswith('multipart/form-data'):
            async with request.form() as form:
                upload = form.get('file')
                if not hasattr(upload, 'read'):
                    raise HTTPException(422, 'CSV upload field must be named file')
                data = await upload.read(MAX_BYTES + 1)
                if len(data) > MAX_BYTES:
                    raise HTTPException(413, 'CSV too large')
                df = pd.read_csv(io.BytesIO(data), nrows=MAX_ROWS+1)
                rows = TypeAdapter(list[Telemetry]).validate_python(df.astype(object).where(pd.notna(df), None).to_dict('records'))
        else:
            raise HTTPException(415, 'Use application/json or multipart/form-data')
    except (ValidationError, ValueError, pd.errors.ParserError) as exc:
        raise HTTPException(422, str(exc)) from exc
    if not 1 <= len(rows) <= MAX_ROWS:
        raise HTTPException(422, 'Provide 1 to 4320 readings')
    return await run_in_threadpool(batch_result, [r.model_dump() for r in rows], loaded)
