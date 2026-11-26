"""Generate reproducible industrial telemetry and controlled fault scenarios."""
from pathlib import Path
import numpy as np
import pandas as pd

DATA_DIR = Path('data')
START = '2026-10-01T00:00:00Z'
# Reference ratings and simulation assumptions; not manufacturer-certified limits.
ASSET_CONFIG = {
    'Chiller-01': {
        'reference_model': 'Carrier AquaForce 30XW Size 200',
        'flow_unit': 'GPM', 'flow_range': (300, 459),
        'supply_temp_range': (6.2, 7.2), 'supply_temp_reference': 6.7,
        'delta_t_range': (4.0, 5.5), 'return_temp_range': (10.2, 12.7),
        'pressure_range': (2.07, 2.76), 'nominal_power_kw': 123.9,
        'cooling_capacity_kw': 672.1, 'standby_power_range': (1., 3.),
        'idle_temp_c': 20., 'idle_pressure_bar': 1.2,
        'notes': 'Constant COP; flow lower bound and temperatures assumed; pressure is a comparable-system assumption.',
    },
    'Compressor-02': {
        'reference_model': 'Atlas Copco ZR 75 VSD-130',
        'flow_unit': 'ACFM', 'flow_range': (159, 413),
        'supply_temp_range': (32., 45.), 'return_temp_range': (25., 35.),
        'pressure_range': (8.5, 9.), 'nominal_power_kw': 93.8,
        'reference_flow': (159, 212, 299, 413),
        'reference_power_kw': (43.3, 52.4, 69.2, 93.8),
        'standby_power_range': (12., 12.),
        'idle_temp_c': 27., 'idle_pressure_bar': .5,
        'notes': 'Interpolated reference power points; temperature and pressure bands assumed.',
    },
    'Pump-03': {
        'reference_model': 'Generic centrifugal cooling-water pump',
        'flow_unit': 'GPM', 'flow_range': (250, 500),
        'return_temp_range': (10., 15.), 'supply_temp_range': (10., 15.3),
        'temperature_rise_range': (0., .3), 'pressure_range': (2.5, 4.2),
        'suction_pressure_bar': 1.2, 'efficiency': .70,
        'nominal_power_kw': 15., 'standby_power_range': (.1, .3),
        'idle_temp_c': 14., 'idle_pressure_bar': 1.2,
        'notes': 'All equipment ranges and efficiency are engineering simulation assumptions.',
    },
}
ASSETS = tuple(ASSET_CONFIG)
GPM_TO_M3_S = .00006309
WATER_DENSITY_KG_M3 = 1000.
WATER_CP_KJ_KG_C = 4.186
BAR_TO_PA = 100000.


def scale_load(bounds, load):
    low, high = bounds
    return low + (high - low) * load

SENSORS = ['power_kw', 'flow_rate', 'supply_temp_c', 'return_temp_c', 'pressure_bar']
COLS = ['timestamp', 'asset_id', *SENSORS, 'status']


def smooth_load(n, rng, phase):
    t = np.arange(n)
    noise = pd.Series(rng.normal(0, 0.07, n)).rolling(15, min_periods=1).mean().to_numpy()
    return np.clip(0.5 + 0.32*np.sin(2*np.pi*t/1440 + phase)
                   + 0.12*np.sin(2*np.pi*t/180 + phase) + noise, 0, 1)


def generate_reference(start, days, seed):
    rng = np.random.default_rng(seed)
    ts = pd.date_range(start, periods=days*1440, freq='min')
    n = len(ts)
    minute = np.arange(n) % 1440
    frames = []
    for i, asset in enumerate(ASSETS):
        load = smooth_load(n, rng, i)
        status = np.where(minute < 120, 'OFF', np.where(minute < 360, 'STANDBY', 'RUNNING'))
        # Continuous operating windows support controlled long-duration events.
        if days >= 14:
            status[11*1440:14*1440] = 'RUNNING'
            status[8*1440:11*1440] = 'RUNNING'
        running = status == 'RUNNING'
        standby = status == 'STANDBY'
        cfg = ASSET_CONFIG[asset]
        flow = scale_load(cfg['flow_range'], load)
        pressure = scale_load(cfg['pressure_range'], load)
        if asset == 'Chiller-01':
            supply = np.clip(cfg['supply_temp_reference'] + rng.normal(0, .12, n),
                             *cfg['supply_temp_range'])
            delta = np.clip(scale_load(cfg['delta_t_range'], load) + rng.normal(0,.05,n),
                            *cfg['delta_t_range'])
            ret = supply + delta
            cop = cfg['cooling_capacity_kw'] / cfg['nominal_power_kw']
            power = flow * GPM_TO_M3_S * WATER_DENSITY_KG_M3 * WATER_CP_KJ_KG_C * delta / cop
        elif asset == 'Compressor-02':
            ret = np.clip(scale_load(cfg['return_temp_range'], load) + rng.normal(0,.15,n),
                          *cfg['return_temp_range'])
            supply = np.clip(ret + 7 + 3*load + rng.normal(0,.15,n), *cfg['supply_temp_range'])
            power = np.interp(flow, cfg['reference_flow'], cfg['reference_power_kw'])
        else:
            ret = np.clip(scale_load(cfg['return_temp_range'], load) + rng.normal(0,.1,n),
                          *cfg['return_temp_range'])
            supply = ret + rng.uniform(*cfg['temperature_rise_range'], size=n)
            power = (flow * GPM_TO_M3_S * (pressure - cfg['suction_pressure_bar'])
                     * BAR_TO_PA / (cfg['efficiency'] * 1000))
        low, high = cfg['standby_power_range']
        idle_power = np.full(n, low) if low == high else rng.uniform(low, high, n)
        idle_temp, idle_pressure = cfg['idle_temp_c'], cfg['idle_pressure_bar']
        # Idle readings represent residual temperature and pressure, not running load.
        power = np.where(running,power,np.where(standby,idle_power,0.))
        flow = np.where(running,flow,0.)
        supply = np.where(running,supply,idle_temp+rng.normal(0,.1,n))
        ret = np.where(running,ret,idle_temp+rng.normal(0,.1,n))
        pressure = np.where(running,pressure,idle_pressure)
        frames.append(pd.DataFrame(dict(timestamp=ts,asset_id=asset,power_kw=power,
            flow_rate=flow,supply_temp_c=supply,return_temp_c=ret,pressure_bar=pressure,status=status)))
    return pd.concat(frames,ignore_index=True)[COLS]


def inject_faults(reference, days, seed, dataset):
    rng = np.random.default_rng(seed)
    raw = reference.copy()
    events = []
    dropped = []
    start = reference.timestamp.min()

    def select(asset, day, hour, minute, count):
        first = start + pd.Timedelta(days=day-1,hours=hour,minutes=minute)
        idx = raw.index[(raw.asset_id == asset) & (raw.timestamp >= first)
                        & (raw.timestamp < first+pd.Timedelta(minutes=count))]
        assert len(idx) == count
        return idx

    def record(idx, kind, fields, event_id, **details):
        rows = raw.loc[idx]
        events.append(dict(dataset=dataset,event_id=event_id,asset_id=rows.asset_id.iloc[0],
            fault_type=kind,start_time=rows.timestamp.iloc[0],end_time=rows.timestamp.iloc[-1],
            affected_records=len(idx),affected_fields=';'.join(fields),**details))

    def missing(event_id, day, hour, count, fields, delete=False):
        idx = select('Pump-03',day,hour,0,count)
        record(idx,'missing_timestamp' if delete else 'missing_values',fields,event_id)
        if delete:
            dropped.extend(idx)
        else:
            raw.loc[idx,fields] = np.nan

    def outlier(asset,day,hour,field,value,event_id):
        idx = select(asset,day,hour,0,1)
        record(idx,'physical_outlier',[field],event_id)
        raw.loc[idx,field] = value

    def operational(asset,day,hour,count,kind,event_id):
        idx = select(asset,day,hour,0,count)
        assert (raw.loc[idx,'status']=='RUNNING').all()
        if kind == 'sensor_freeze':
            field = 'return_temp_c'
            record(idx,kind,[field],event_id)
            raw.loc[idx,field] = raw.loc[idx[0],field]
        else:
            # Hold the operating point fixed in both datasets; only power changes.
            point = reference.loc[idx[0],SENSORS].astype(float).to_numpy()
            fixed = np.tile(point,(count,1))
            # Retain small common-mode temperature variation to avoid an accidental freeze.
            variation = rng.uniform(-.015,.015,count)
            fixed[:,2] += variation
            fixed[:,3] += variation
            reference.loc[idx,SENSORS] = fixed
            raw.loc[idx,SENSORS] = fixed
            increase = .25 if kind == 'efficiency_drift' else float(rng.uniform(.35,.45))
            record(idx,kind,['power_kw'],event_id,relative_power_increase=increase)
            factor = np.linspace(1,1+increase,count) if kind == 'efficiency_drift' else 1+increase
            raw.loc[idx,'power_kw'] = point[0]*factor

    if days >= 14:
        for args in [('M01',1,10,1,['power_kw']),('M02',1,14,10,['power_kw']),
                     ('M03',2,11,10,['power_kw','flow_rate']),
                     ('M04',2,15,15,['supply_temp_c','return_temp_c']),
                     ('M05',3,9,20,['power_kw','flow_rate','pressure_bar']),
                     ('M06',3,14,25,SENSORS),('M07',4,10,30,COLS,True)]:
            missing(*args)
        outlier('Chiller-01',5,10,'power_kw',-10,'O01')
        outlier('Compressor-02',6,12,'supply_temp_c',999,'O02')
        # Each held-out split includes all three operational fault categories.
        for day, prefix in [(9,'V'),(12,'T')]:
            operational('Chiller-01',day,8,20,'power_surge',prefix+'01')
            operational('Compressor-02',day,10,60,'sensor_freeze',prefix+'02')
            operational('Pump-03',day,0,2881,'efficiency_drift',prefix+'03')
    else:
        missing('B01',1,8,15,['power_kw'])
        missing('B02',1,9,20,COLS,True)
        outlier('Chiller-01',1,10,'supply_temp_c',999,'B03')
        operational('Chiller-01',1,12,20,'power_surge','B04')
        operational('Compressor-02',1,14,60,'sensor_freeze','B05')
        outlier('Pump-03',1,16,'power_kw',-10,'B06')
    return raw.drop(index=dropped).reset_index(drop=True),pd.DataFrame(events)


def save_csv(frame, name):
    frame.to_csv(DATA_DIR/name,index=False,date_format='%Y-%m-%dT%H:%M:%SZ')


def main():
    DATA_DIR.mkdir(exist_ok=True)
    reference = generate_reference(START,14,42)
    raw, manifest = inject_faults(reference,14,43,'main')
    batch_reference = generate_reference('2026-10-15T00:00:00Z',1,142)
    batch, batch_manifest = inject_faults(batch_reference,1,143,'batch')
    assert len(reference)==60480 and len(raw)==60450
    assert len(batch)==4300
    for frame in (reference,raw,batch):
        assert list(frame.columns)==COLS
        assert not frame.duplicated(['asset_id','timestamp']).any()
    save_csv(reference,'reference_telemetry.csv')
    save_csv(raw,'raw_telemetry.csv')
    save_csv(batch,'sample_batch_24h.csv')
    save_csv(pd.concat([manifest,batch_manifest],ignore_index=True),'injection_manifest.csv')
    print('Reference rows:',len(reference))
    print('Raw rows:',len(raw))
    print('Batch rows:',len(batch))
    print('Main fault events:',len(manifest))
    print('Batch fault events:',len(batch_manifest))
    print(raw.groupby(['asset_id','status']).size().to_string())
    print('Missing sensor values:',raw[SENSORS].isna().sum().to_dict())
    print('CSV files saved to',DATA_DIR.resolve())


if __name__ == '__main__':
    main()
