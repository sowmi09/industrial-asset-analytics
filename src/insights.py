"""Physics-based energy audit and diagnostic heuristics; no model fitting.
Run from repository root: python src/insights.py
Inputs are minute-aligned cleaned telemetry and trusted local model bundles.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import sklearn

if __package__:
    from .generate_data import (ASSET_CONFIG, GPM_TO_M3_S, WATER_DENSITY_KG_M3,
                                WATER_CP_KJ_KG_C, BAR_TO_PA)
    from .train_eval import score_readings, flag, FEATURES, POLICY
else:
    from generate_data import (ASSET_CONFIG, GPM_TO_M3_S, WATER_DENSITY_KG_M3,
                               WATER_CP_KJ_KG_C, BAR_TO_PA)
    from train_eval import score_readings, flag, FEATURES, POLICY

TARIFF_USD = 0.12
MINUTES_PER_READING = 1
SURGE_PCT = 20.0  # Proposed simulation-specific tolerance, not a safety limit.
DRIFT_PCT = 10.0  # Minimum increase across 24h, not a fixed 25% diagnosis.
MINUTE = pd.Timedelta(minutes=1)
ACTIONS = {
    'Chiller-01': ('Inspect motor load and condenser conditions.',
                   'Inspect heat exchanger fouling and refrigerant circuit.'),
    'Compressor-02': ('Check motor load or compressor stage bypass.',
                      'Inspect seals, air leakage and compression efficiency.'),
    'Pump-03': ('Inspect motor loading and hydraulic resistance.',
                'Inspect impeller wear, blockages and hydraulic efficiency.'),
}


def load_bundles(directory='models'):
    """Load only trusted artifacts produced by the existing training script."""
    bundles = {}
    for asset, (algorithm, wp, cp) in POLICY.items():
        path = Path(directory) / f'{asset}.joblib'
        b = joblib.load(path)
        if (b['asset_id'] != asset or b['algorithm'] != algorithm
                or b['features'] != FEATURES or b['sklearn_version'] != sklearn.__version__
                or b['score_policy'] != 'training_percentile_anchors_v2'
                or b['warning_percentile'] != wp or b['critical_percentile'] != cp):
            raise ValueError(f'Incompatible bundle: {path}')
        bundles[asset] = b
    return bundles


def reliable(g, fields):
    """Reject missing, invalid or imputed inputs relevant to this calculation."""
    mask = g.status.eq('RUNNING') & ~flag(g, 'is_missing_timestamp')
    mask &= np.isfinite(g[fields].to_numpy(float)).all(axis=1)
    for col in ('invalid_fields', 'imputed_fields', 'missing_fields'):
        mask &= ~g[col].fillna('').astype(str).map(lambda s: bool(set(s.split('|')) & set(fields)))
    return mask


def expected_power(g):
    """Return expected kW, NaN when operating conditions are unsupported."""
    asset = g.asset_id.iloc[0]
    cfg = ASSET_CONFIG[asset]
    q = g.flow_rate
    fields = ['flow_rate', 'pressure_bar']
    if asset == 'Chiller-01':
        fields += ['supply_temp_c', 'return_temp_c']
    valid = reliable(g, fields) & q.between(*cfg['flow_range'])
    valid &= g.pressure_bar.between(*cfg['pressure_range'])
    if asset == 'Chiller-01':
        valid &= (g.delta_t.between(*cfg['delta_t_range'])
                  & ~flag(g, 'is_supply_temp_frozen') & ~flag(g, 'is_return_temp_frozen'))
        cop = cfg['cooling_capacity_kw'] / cfg['nominal_power_kw']
        power = q * GPM_TO_M3_S * WATER_DENSITY_KG_M3 * WATER_CP_KJ_KG_C * g.delta_t / cop
    elif asset == 'Compressor-02':
        power = pd.Series(np.interp(q, cfg['reference_flow'], cfg['reference_power_kw']), index=g.index)
    else:
        power = q * GPM_TO_M3_S * (g.pressure_bar - cfg['suction_pressure_bar']) * BAR_TO_PA / (cfg['efficiency'] * 1000)
    return power.where(valid & np.isfinite(power) & power.gt(0))


def runs(g, mask):
    """Yield contiguous minute groups; NORMAL, missing slots and gaps break a run."""
    subset = g.loc[mask]
    if subset.empty:
        return
    ids = subset.timestamp.diff().ne(MINUTE).cumsum()
    for _, part in subset.groupby(ids, sort=False):
        yield part


def freeze_mask(g, sensor, retrospective=True):
    """Require identical reliable readings spanning strictly over 30 minutes."""
    out = pd.Series(False, index=g.index)
    valid = reliable(g, [sensor])
    # Split on invalid readings, timestamp gaps and changes in temperature.
    boundary = (~valid | ~valid.shift(fill_value=False) | g.timestamp.diff().ne(MINUTE)
                | g[sensor].ne(g[sensor].shift()))
    for _, part in g.loc[valid].groupby(boundary.cumsum()[valid], sort=False):
        elapsed = part.timestamp - part.timestamp.iloc[0]
        confirmed = elapsed > pd.Timedelta(minutes=30)
        if confirmed.any():
            out.loc[part.index] = True if retrospective else confirmed
    return out


def degradation_mask(g, retrospective=True):
    """Detect a sustained increase in baseline-relative specific energy.

    Require five one-hour means spaced six hours apart (24 hours between
    the first and last means), >=10 percentage-point increase and a
    positive rise at each step. A step jump alone is insufficient.
    """
    out = pd.Series(False, index=g.index)
    valid = g.efficiency_deviation_pct.notna()
    for _, part in enumerate(runs(g, valid)):
        x = part.efficiency_deviation_pct
        if len(x) < 1500:
            continue
        avg = x.rolling(60, min_periods=60).mean()
        a, b, c, d, e = avg.shift(1440), avg.shift(1080), avg.shift(720), avg.shift(360), avg
        # Five means must rise; reject isolated spikes/one-off steps.
        hit = ((b-a >= 0.5) & (c-b >= 0.5) & (d-c >= 0.5) & (e-d >= 0.5)
               & (e-a >= DRIFT_PCT) & (e >= DRIFT_PCT) & (x >= e-2.0))
        hit &= (part.timestamp - part.timestamp.iloc[0]) >= pd.Timedelta(hours=24)
        if retrospective:
            hit = hit.iloc[::-1].rolling(1500, min_periods=1).max().iloc[::-1].astype(bool)
        out.loc[part.index] = hit
    return out


def enrich_asset(scored, retrospective=True):
    """Attach energy estimates and independent diagnostic evidence to scored rows."""
    g = scored.sort_values('timestamp').reset_index(drop=True).copy()
    if g.empty or g.asset_id.nunique() != 1:
        raise ValueError('Expected nonempty telemetry for one asset')
    g.timestamp = pd.to_datetime(g.timestamp, utc=True, errors='raise')
    if g.timestamp.duplicated().any() or not g.timestamp.eq(g.timestamp.dt.floor('min')).all():
        raise ValueError('Expected unique minute-aligned timestamps')
    g['expected_power_kw'] = expected_power(g)
    actual_ok = reliable(g, ['power_kw']) & g.power_kw.ge(0)
    g['excess_power_kw'] = (g.power_kw - g.expected_power_kw).where(actual_ok)
    # Eliminate floating-point cancellation noise, not meaningful negative differences.
    g['excess_power_kw'] = g.excess_power_kw.mask(g.excess_power_kw.abs() < 1e-9, 0.0)
    g['excess_energy_kwh'] = g.excess_power_kw * MINUTES_PER_READING / 60
    g['baseline_specific_energy'] = g.expected_power_kw / (g.flow_rate + 1e-5)
    g['efficiency_deviation_pct'] = ((g.specific_energy - g.baseline_specific_energy)
                                      / g.baseline_specific_energy * 100).where(actual_ok)
    # Independent freeze diagnostics refine the preprocessing flag's 30-sample rule.
    g['supply_freeze_detected'] = freeze_mask(g, 'supply_temp_c', retrospective)
    g['return_freeze_detected'] = freeze_mask(g, 'return_temp_c', retrospective)
    g['degradation_detected'] = degradation_mask(g, retrospective)
    g['high_power_detected'] = g.efficiency_deviation_pct.gt(SURGE_PCT)
    g['ml_detected'] = g.classification.isin(['WARNING', 'CRITICAL'])
    g['rule_detected'] = g[['supply_freeze_detected', 'return_freeze_detected',
                            'degradation_detected', 'high_power_detected']].any(axis=1)
    g['detected'] = g.ml_detected | g.rule_detected
    return g


def diagnosis(part):
    asset = part.asset_id.iloc[0]
    messages, actions = [], []
    if part.high_power_detected.any():
        messages.append('High power consumption detected without proportional flow/operating-load increase.')
        actions.append(ACTIONS[asset][0])
    for sensor in ('supply', 'return'):
        if part[f'{sensor}_freeze_detected'].any():
            messages.append(f'{sensor.title()} temperature sensor freeze detected (>30 minutes RUNNING). Possible probe failure or communication lockup.')
            actions.append('Inspect temperature probe and communication connections.')
    if part.degradation_detected.any():
        peak = part.loc[part.degradation_detected, 'efficiency_deviation_pct'].max()
        messages.append(f'Gradual efficiency loss detected over 24+ hours (peak {peak:.1f}% above baseline).')
        actions.append(ACTIONS[asset][1])
    if not messages:
        messages.append('ML detected unusual telemetry; no specific diagnostic heuristic confirmed.')
        actions.append('Review sensor quality and operating conditions before maintenance.')
    return ' '.join(messages), ' '.join(dict.fromkeys(actions))


def summarize_periods(g):
    """Union of ML/rule alerts, grouped per asset; each minute counted once."""
    rows = []
    for part in runs(g, g.detected):
        energy = part.excess_energy_kwh
        net = energy.sum(min_count=1)
        positive = energy.clip(lower=0).sum(min_count=1)
        msg, action = diagnosis(part)
        severity = ('CRITICAL' if part.classification.eq('CRITICAL').any() else
                    'WARNING' if part.ml_detected.any() else 'RULE_ONLY')
        # Keep the signed PDF result and a separate positive-only waste metric.
        rows.append(dict(asset_id=part.asset_id.iloc[0], start_utc=part.timestamp.iloc[0],
                         end_utc=part.timestamp.iloc[-1], duration_minutes=len(part),
                         ml_severity=severity, warning_minutes=int(part.classification.eq('WARNING').sum()),
                         critical_minutes=int(part.classification.eq('CRITICAL').sum()),
                         rule_only_minutes=int((part.rule_detected & ~part.ml_detected).sum()),
                         energy_evaluated_minutes=int(energy.notna().sum()),
                         energy_unavailable_minutes=int(energy.isna().sum()),
                         net_excess_energy_kwh=net, net_energy_cost_usd=net*TARIFF_USD,
                         estimated_cost_loss_usd=max(net, 0)*TARIFF_USD if pd.notna(net) else np.nan,
                         positive_excess_energy_kwh=positive,
                         mean_load_ratio=part.load_ratio.mean(),
                         peak_rolling_power_mean_1h_kw=part.rolling_power_mean_1h.max(),
                         diagnostic_message=msg, recommended_action=action))
    return rows


def audit(cleaned, bundles, retrospective=True):
    """Reusable API interface: return enriched readings and incident summary."""
    if cleaned.empty or not cleaned.asset_id.isin(bundles).all():
        raise ValueError('Empty telemetry or unknown asset')
    frames, periods = [], []
    for asset, group in cleaned.groupby('asset_id', sort=True):
        scored = score_readings(group, bundles[asset])
        enriched = enrich_asset(scored, retrospective)
        frames.append(enriched)
        periods.extend(summarize_periods(enriched))
    columns = ['asset_id', 'start_utc', 'end_utc', 'duration_minutes', 'ml_severity',
               'warning_minutes', 'critical_minutes', 'rule_only_minutes',
               'energy_evaluated_minutes', 'energy_unavailable_minutes',
               'net_excess_energy_kwh', 'net_energy_cost_usd', 'estimated_cost_loss_usd',
               'positive_excess_energy_kwh', 'mean_load_ratio',
               'peak_rolling_power_mean_1h_kw', 'diagnostic_message', 'recommended_action']
    return pd.concat(frames, ignore_index=True), pd.DataFrame(periods, columns=columns)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', default='data/cleaned_telemetry.csv')
    p.add_argument('--models', default='models')
    p.add_argument('--output', default='outputs/insights')
    a = p.parse_args()
    df = pd.read_csv(a.input, low_memory=False)
    df.timestamp = pd.to_datetime(df.timestamp, utc=True, errors='raise')
    readings, periods = audit(df, load_bundles(a.models))
    out = Path(a.output); out.mkdir(parents=True, exist_ok=True)
    periods.to_csv(out / 'all_incidents.csv', index=False)
    summary = []
    for asset, g in readings.groupby('asset_id'):
        detected = g[g.detected]
        e = detected.excess_energy_kwh
        net = e.sum(min_count=1) if len(e) else 0.0
        asset_periods = periods[periods.asset_id.eq(asset)] if not periods.empty else periods
        cost = asset_periods.estimated_cost_loss_usd.sum(min_count=1) if len(asset_periods) else 0.0
        summary.append(dict(asset_id=asset, records=len(g), ml_alert_minutes=int(g.ml_detected.sum()),
                            rule_only_minutes=int((g.rule_detected & ~g.ml_detected).sum()),
                            freeze_minutes=int((g.supply_freeze_detected | g.return_freeze_detected).sum()),
                            degradation_minutes=int(g.degradation_detected.sum()),
                            high_power_minutes=int(g.high_power_detected.sum()),
                            incidents=len(asset_periods), energy_evaluated_minutes=int(e.notna().sum()),
                            energy_unavailable_minutes=int(e.isna().sum()), net_excess_energy_kwh=net,
                            estimated_cost_loss_usd=cost))
    summary = pd.DataFrame(summary)
    summary.to_csv(out / 'all_summary.csv', index=False)
    print(summary.to_string(index=False))
    print(f'Saved incident and summary CSVs in {out}. No training or recalibration.')


if __name__ == '__main__':
    main()
