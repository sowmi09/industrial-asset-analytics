"""Validate minute telemetry, retain quality provenance and derive features."""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd

try:
    from .generate_data import ASSET_CONFIG, SENSORS, COLS
except ImportError:
    from generate_data import ASSET_CONFIG, SENSORS, COLS

MODEL_FEATURES = [*SENSORS, 'delta_t', 'specific_energy', 'load_ratio']
# Public cleaned telemetry schema; intermediate flags stay internal.
CLEANED_COLUMNS = [
    *COLS,
    'delta_t', 'specific_energy', 'rolling_power_mean_1h', 'load_ratio',
    'is_missing_timestamp', 'original_missing_fields', 'invalid_fields',
    'imputed_fields', 'missing_fields', 'is_supply_temp_frozen',
    'is_return_temp_frozen', 'is_rolling_power_estimated', 'is_ml_eligible',
]

STATUSES = {'RUNNING', 'STANDBY', 'OFF'}
MAX_GAP = 10
# Conservative simulation-specific tolerances, not equipment safety limits.
REL_TOL = 0.05
TEMP_TOL = 0.5


def validate(raw):
    if not set(COLS).issubset(raw.columns) or raw.empty:
        raise ValueError('Required telemetry fields missing or input empty')
    df = raw[COLS].copy()
    if df.asset_id.isna().any() or not df.asset_id.isin(ASSET_CONFIG).all():
        raise ValueError('Unknown or missing asset identifier')
    # Reject timezone-naive strings rather than silently interpreting as UTC.
    ts = df.timestamp.astype(str)
    if not ts.str.endswith(('Z', '+00:00', '+0000')).all():
        raise ValueError('Timestamps must explicitly specify UTC')
    df.timestamp = pd.to_datetime(df.timestamp, utc=True, errors='raise')
    if not df.timestamp.eq(df.timestamp.dt.floor('min')).all():
        raise ValueError('Timestamps must align to minute boundaries')
    if df.duplicated(['asset_id', 'timestamp']).any():
        raise ValueError('Duplicate asset/timestamp keys require resolution')
    if not df.status.dropna().isin(STATUSES).all():
        raise ValueError('Unknown operating status')
    for col in SENSORS:
        df[col] = pd.to_numeric(df[col], errors='raise').astype(float)
    return df.sort_values(['asset_id', 'timestamp'])


def field_names(flags):
    return flags.apply(lambda row: '|'.join(row.index[row.to_numpy(dtype=bool)]), axis=1)


def stable(values, col):
    if not np.isfinite(values).all():
        return False
    spread = np.ptp(values)
    if col.endswith('_temp_c'):
        return spread <= TEMP_TOL
    return spread <= REL_TOL * max(abs(np.median(values)), 1e-6)


def process_asset(group, start, end, retrospective=True, boundaries=()):
    asset = group.asset_id.iloc[0]
    g = group.set_index('timestamp').reindex(pd.date_range(start, end, freq='min', name='timestamp'))
    inserted = g.asset_id.isna()
    g['asset_id'] = asset
    original_missing = g[SENSORS].isna() & ~inserted.to_numpy()[:, None]
    invalid = pd.DataFrame(False, index=g.index, columns=SENSORS)
    for col in SENSORS:
        x = g[col]
        bad = x.notna() & ~np.isfinite(x)
        if col in ('power_kw', 'flow_rate'):
            bad |= x.lt(0)
        elif col.endswith('_temp_c'):
            bad |= x.lt(-5) | x.gt(110)
        invalid[col] = bad
        g.loc[bad, col] = np.nan
    frozen = pd.DataFrame(False, index=g.index, columns=SENSORS)
    for col in ('supply_temp_c', 'return_temp_c'):
        x = g[col].where(g.status.eq('RUNNING'))
        trailing = x.rolling(30, min_periods=30).std().eq(0)
        # Also require exact equality to guard against floating-point cancellation.
        trailing |= x.rolling(30, min_periods=30).max().eq(x.rolling(30, min_periods=30).min())
        trailing &= x.rolling(30, min_periods=30).max().eq(x.rolling(30, min_periods=30).min())
        g['is_' + col + '_freeze_window'] = trailing
        if retrospective:
            # Mark the full confirmed interval, including the first 29 readings.
            frozen[col] = trailing.iloc[::-1].rolling(30, min_periods=1).max().iloc[::-1].astype(bool)
        else:
            frozen[col] = trailing
    observed = g[SENSORS].copy()
    imputed = pd.DataFrame(False, index=g.index, columns=SENSORS)
    partition = np.searchsorted(np.array([pd.Timestamp(b).value for b in boundaries], dtype='int64'), g.index.asi8, side='right')
    if retrospective:
        for col in SENSORS:
            missing = observed[col].isna().to_numpy()
            edges = np.diff(np.r_[False, missing, False].astype(int))
            for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
                # b is exclusive. Require observed bounds and a single missing channel.
                if b-a > MAX_GAP or a == 0 or b == len(g) or partition[a-1] != partition[b]:
                    continue
                window = slice(a-1, b+1)
                status = g.status.iloc[window]
                if status.isna().any() or status.nunique() != 1 or inserted.iloc[window].any():
                    continue
                if observed.iloc[a:b].isna().sum(axis=1).ne(1).any():
                    continue
                if invalid.iloc[a:b][col].any() or frozen.iloc[window].any(axis=None):
                    continue
                supporting = [c for c in SENSORS if c != col]
                if not all(stable(observed[c].iloc[window].to_numpy(), c) for c in supporting):
                    continue
                endpoints = observed[col].iloc[[a-1,b]].to_numpy()
                if not stable(endpoints, col):
                    continue
                g.iloc[a:b, g.columns.get_loc(col)] = np.linspace(*endpoints, b-a+2)[1:-1]
                imputed.iloc[a:b, imputed.columns.get_loc(col)] = True
    unresolved = g[SENSORS].isna()
    g['is_missing_timestamp'] = inserted
    g['original_missing_fields'] = field_names(original_missing)
    g['invalid_fields'] = field_names(invalid)
    g['imputed_fields'] = field_names(imputed)
    g['missing_fields'] = field_names(unresolved)
    g['is_imputed'] = imputed.any(axis=1)
    g['has_missing_values'] = unresolved.any(axis=1) | g.status.isna()
    g['is_invalid_power'] = invalid.power_kw
    g['is_invalid_temperature'] = invalid[['supply_temp_c','return_temp_c']].any(axis=1)
    g['is_supply_temp_frozen'] = frozen.supply_temp_c
    g['is_return_temp_frozen'] = frozen.return_temp_c
    g['is_sensor_frozen'] = frozen.any(axis=1)
    g['delta_t'] = (g.return_temp_c - g.supply_temp_c).mask(g.is_sensor_frozen)
    g['specific_energy'] = (g.power_kw / (g.flow_rate + 1e-5)).where(g.flow_rate.gt(0))
    g['rolling_power_mean_1h'] = g.power_kw.rolling(60, min_periods=60).mean()
    g['load_ratio'] = g.power_kw / ASSET_CONFIG[asset]['nominal_power_kw']
    g['is_delta_t_estimated'] = imputed[['supply_temp_c','return_temp_c']].any(axis=1) & g.delta_t.notna()
    g['is_specific_energy_estimated'] = imputed[['power_kw','flow_rate']].any(axis=1) & g.specific_energy.notna()
    g['is_load_ratio_estimated'] = imputed.power_kw & g.load_ratio.notna()
    g['is_rolling_power_estimated'] = imputed.power_kw.rolling(60, min_periods=1).max().astype(bool) & g.rolling_power_mean_1h.notna()
    g['is_ml_eligible'] = (g.status.eq('RUNNING') & ~g.is_imputed & ~invalid.any(axis=1) & ~g.is_sensor_frozen & np.isfinite(g[MODEL_FEATURES]).all(axis=1))
    return g.reset_index()


def process_telemetry(raw, start=None, end=None, retrospective=True, boundaries=()):
    """Process raw readings only. Optional start/end are inclusive UTC minute slots."""
    df = validate(raw)
    start = df.timestamp.min() if start is None else pd.to_datetime(start, utc=True)
    end = df.timestamp.max() if end is None else pd.to_datetime(end, utc=True)
    if start > df.timestamp.min() or end < df.timestamp.max() or start > end:
        raise ValueError('Observation bounds must contain input timestamps')
    if start != start.floor('min') or end != end.floor('min'):
        raise ValueError('Observation bounds must align to minutes')
    return pd.concat([process_asset(g, start, end, retrospective, boundaries) for _, g in df.groupby('asset_id')], ignore_index=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', default='data/raw_telemetry.csv')
    parser.add_argument('--output', default='data/cleaned_telemetry.csv')
    args = parser.parse_args()
    raw = pd.read_csv(args.input)
    # Prevent retrospective interpolation across chronological split boundaries.
    boundaries = ('2026-10-09T00:00:00Z','2026-10-12T00:00:00Z') if args.input == 'data/raw_telemetry.csv' else ()
    clean = process_telemetry(raw, boundaries=boundaries)
    path = Path(args.output)
    if path.resolve() == Path(args.input).resolve():
        raise ValueError('Output must not overwrite raw input')
    path.parent.mkdir(parents=True, exist_ok=True)
    clean.to_csv(path, columns=CLEANED_COLUMNS, index=False, date_format='%Y-%m-%dT%H:%M:%SZ')
    print('Input rows:', len(raw), '| Cleaned rows:', len(clean))
    for col in ['is_missing_timestamp','is_imputed','has_missing_values','is_invalid_power','is_invalid_temperature','is_sensor_frozen','is_ml_eligible']:
        print(col + ':', int(clean[col].sum()))
    print('Remaining sensor NaNs:', clean[SENSORS].isna().sum().to_dict())
    print('Exported columns:', len(CLEANED_COLUMNS))
    print('Saved:', path)


if __name__ == '__main__':
    main()
