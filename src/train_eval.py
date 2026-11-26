"""Asset-specific anomaly detection, calibration, validation and reporting."""
import argparse
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Patch

WARNING_COLOR = "#F59E0B"
CRITICAL_COLOR = "#C81E1E"

FEATURES = ['power_kw', 'flow_rate', 'supply_temp_c', 'return_temp_c', 'pressure_bar']
ASSETS = ('Chiller-01', 'Compressor-02', 'Pump-03')
POLICY = {'Chiller-01': ('OneClassSVM', 99., 99.9),
          'Compressor-02': ('OneClassSVM', 99., 99.9),
          'Pump-03': ('IsolationForest', 95., 99.)}
START, VAL, TEST, END = map(lambda x: pd.Timestamp(x, tz='UTC'),
                           ['2026-10-01', '2026-10-09', '2026-10-12', '2026-10-15'])
TARGETS = [0., .5, .75, 1.]


def flag(frame, name):
    return frame[name].astype(str).str.strip().str.lower().isin(['true', '1'])


def eligible(frame):
    return (frame.status.eq('RUNNING') & np.isfinite(frame[FEATURES].to_numpy(float)).all(axis=1)
            & ~flag(frame, 'is_missing_timestamp')
            & frame.invalid_fields.fillna('').astype(str).str.strip().eq('')
            & frame.imputed_fields.fillna('').astype(str).str.strip().eq(''))


def raw_score(model, algorithm, x):
    return -(model.score_samples(x) if algorithm == 'IsolationForest' else model.decision_function(x)).ravel()


def classify(score):
    return np.select([score >= .75, score >= .5], ['CRITICAL', 'WARNING'], default='NORMAL')


def score_readings(frame, bundle):
    """Score validated telemetry (single or batch); no fitting or temporal feature calculation.

    The caller must validate raw inputs and supply quality flags via preprocessing.
    Frozen numeric measurements are scored provisionally; freezing is a separate temporal rule.
    """
    g = frame.copy()
    if not g.asset_id.eq(bundle['asset_id']).all():
        raise ValueError('Telemetry asset does not match model bundle')
    mask = eligible(g)
    frozen = flag(g, 'is_supply_temp_frozen') | flag(g, 'is_return_temp_frozen')
    g['algorithm'] = bundle['algorithm']
    g['sensor_frozen'] = frozen
    g['prediction_status'] = np.where(mask, np.where(frozen, 'PROVISIONAL', 'SCORED'), 'NOT_EVALUATED')
    g['classification'] = 'NOT_EVALUATED'
    for col in ['native_prediction', 'native_decision', 'raw_anomaly_score', 'anomaly_score']:
        g[col] = np.nan
    if mask.any():
        x = bundle['scaler'].transform(g.loc[mask, FEATURES])
        m = bundle['model']
        raw = raw_score(m, bundle['algorithm'], x)
        s = np.interp(raw, bundle['calibration_anchors'], TARGETS)
        g.loc[mask, 'native_prediction'] = m.predict(x)
        g.loc[mask, 'native_decision'] = m.decision_function(x).ravel()
        g.loc[mask, 'raw_anomaly_score'] = raw
        g.loc[mask, 'anomaly_score'] = s
        g.loc[mask, 'classification'] = classify(s)
    return g


def fit_asset(train, asset, algorithm, warning_pct, critical_pct):
    scaler = StandardScaler().fit(train[FEATURES])
    x = scaler.transform(train[FEATURES])
    model = (IsolationForest(n_estimators=200, max_samples=1024, contamination='auto',
                             random_state=42, n_jobs=2) if algorithm == 'IsolationForest'
             else OneClassSVM(kernel='rbf', nu=.01, gamma='scale'))
    model.fit(x)
    raw = raw_score(model, algorithm, x)
    cap = 1. if algorithm == 'IsolationForest' else float(model.offset_[0])
    anchors = np.array([raw.min(), *np.percentile(raw, [warning_pct, critical_pct]), cap])
    if not np.isfinite(anchors).all() or not (np.diff(anchors) > 0).all():
        raise ValueError(f'{asset}/{algorithm}: tied or invalid calibration anchors {anchors}; review score distribution')
    return dict(asset_id=asset, algorithm=algorithm, model=model, scaler=scaler, features=FEATURES.copy(),
                calibration_anchors=anchors.tolist(), calibration_targets=TARGETS,
                warning_percentile=warning_pct, critical_percentile=critical_pct,
                sorted_training_scores=np.sort(raw), training_records=len(train),
                sklearn_version=sklearn.__version__, score_policy='training_percentile_anchors_v2')


def score_figure(b, g, path):
    """Validation native score and normalized score; anchors fitted on training only."""
    fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True, constrained_layout=True)
    a = b['calibration_anchors']; alg = b['algorithm']
    native = -g.raw_anomaly_score
    ax = axes[0]
    ax.plot(g.timestamp, native, color='#2563eb', lw=.8, label='Original model score')
    for j, pct, color in [(1,b['warning_percentile'],WARNING_COLOR),(2,b['critical_percentile'],CRITICAL_COLOR)]:
        ax.axhline(-a[j], color=color, ls='--', label=f'Training reversed P{pct:g} = {a[j]:.5g} (original {-a[j]:.5g})')
    ax.set_ylabel('score_samples()' if alg == 'IsolationForest' else 'decision_function()')
    ax.set_title('Original validation scores — lower means more anomalous', loc='left')
    ax.legend(loc='best', fontsize=9)
    ax = axes[1]
    draw_score(ax,g)
    ax.set_title('Normalized anomaly score — higher means more anomalous',loc='left')
    fig.suptitle(f'{b["asset_id"]} | {alg} | Validation', fontsize=16)
    fig.savefig(path,dpi=160); plt.close(fig)


def draw_score(ax,g):
    ax.axhspan(0,.5,color='#16a34a',alpha=.06)
    ax.axhspan(.5,.75,color=WARNING_COLOR,alpha=.15)
    ax.axhspan(.75,1,color=CRITICAL_COLOR,alpha=.12)
    ax.plot(g.timestamp,g.anomaly_score,color='#4338ca',lw=.8)
    for t,c,label in [(.5,WARNING_COLOR,'WARNING threshold (0.50)'),(.75,CRITICAL_COLOR,'CRITICAL threshold (0.75)')]:
        ax.axhline(t,color=c,ls='--',lw=1.2,label=label)
    ax.legend(loc='upper right',fontsize=8)
    ax.set(ylim=(0,1.02),ylabel='Anomaly score',xlabel='UTC')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d %H:%M',tz=mdates.UTC))
    ax.grid(alpha=.15)


def timeline(ax, b, g):
    dual = b['asset_id'] != 'Compressor-02'
    cols = ('power_kw','flow_rate') if dual else ('supply_temp_c','return_temp_c')
    labels = ('Power (kW)','Flow (GPM)') if dual else ('Supply temperature (°C)','Return temperature (°C)')
    other = ax.twinx() if dual else ax
    l1, = ax.plot(g.timestamp,g[cols[0]],color='#2563eb',lw=.85,label=labels[0])
    l2, = other.plot(g.timestamp,g[cols[1]],color='#0d9488',lw=.85,label=labels[1])
    ax.set_ylabel(labels[0] if dual else 'Temperature (°C)')
    if dual: other.set_ylabel(labels[1])
    # Highlight whole time intervals, not individual sensor curves. Break at gaps
    # and at every classification change; one telemetry sample represents one minute.
    ordered = g.sort_values('timestamp')
    groups = (ordered.classification.ne(ordered.classification.shift()) |
              ordered.timestamp.diff().ne(pd.Timedelta(minutes=1))).cumsum()
    for _, event in ordered.groupby(groups, sort=False):
        state = event.classification.iloc[0]
        if state not in ('WARNING', 'CRITICAL'):
            continue
        color = WARNING_COLOR if state == 'WARNING' else CRITICAL_COLOR
        left = event.timestamp.iloc[0] - pd.Timedelta(seconds=30)
        right = event.timestamp.iloc[-1] + pd.Timedelta(seconds=30)
        ax.axvspan(left, right, facecolor=color,
                   alpha=.26 if state == 'WARNING' else .34,
                   linewidth=0, zorder=0)
    handles = [l1, l2, Patch(facecolor=WARNING_COLOR, alpha=.55, label='WARNING interval'),
               Patch(facecolor=CRITICAL_COLOR, alpha=.65, label='CRITICAL interval')]
    ax.legend(handles=handles,loc='upper right',fontsize=8,ncol=2)
    ax.grid(alpha=.15)


def timeline_figure(b,g,path):
    fig, axes=plt.subplots(2,1,figsize=(13,7),sharex=True,constrained_layout=True)
    timeline(axes[0],b,g); draw_score(axes[1],g)
    axes[0].set_title('Actual sensor measurements | shaded model-alert intervals',loc='left')
    fig.suptitle(f'{b["asset_id"]} | {b["algorithm"]} | Validation',fontsize=16)
    fig.savefig(path,dpi=160); plt.close(fig)


def metrics(y, pred):
    tp=int((y&pred).sum()); fp=int((~y&pred).sum()); fn=int((y&~pred).sum()); tn=int((~y&~pred).sum())
    p=tp/(tp+fp) if tp+fp else 0.; r=tp/(tp+fn) if tp+fn else 0.
    return dict(tp=tp,fp=fp,fn=fn,tn=tn,precision=p,recall=r,f1=2*p*r/(p+r) if p+r else 0.)


def verify(results, manifest):
    """Independent reference used only after all predictions are fixed."""
    if not manifest.exists():
        print('Reference events unavailable: labelled evaluation skipped.')
        return None
    events=pd.read_csv(manifest)
    events=events[events.dataset.eq('main') & events.fault_type.isin(['power_surge','sensor_freeze','efficiency_drift'])].copy()
    for c in ['start_time','end_time']: events[c]=pd.to_datetime(events[c],utc=True)
    labels=[]
    for g in results:
        y=pd.Series(False,index=g.index)
        for e in events[events.asset_id.eq(g.asset_id.iloc[0])].itertuples():
            y |= g.timestamp.between(e.start_time,e.end_time)
        labels.append(y)
    return labels


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',default='data/cleaned_telemetry.csv'); p.add_argument('--models',default='models')
    p.add_argument('--output',default='outputs/model_evaluation'); p.add_argument('--manifest',default='data/injection_manifest.csv')
    p.add_argument('--score-test',action='store_true',help='Use saved models without fitting; run only after approval')
    a=p.parse_args()
    out=Path(a.output); models=Path(a.models); out.mkdir(parents=True,exist_ok=True); models.mkdir(parents=True,exist_ok=True)
    df=pd.read_csv(a.input,low_memory=False); df.timestamp=pd.to_datetime(df.timestamp,utc=True,errors='raise')
    if df.duplicated(['asset_id','timestamp']).any(): raise ValueError('Duplicate timestamps')
    split='test' if a.score_test else 'validation'; lo,hi=(TEST,END) if a.score_test else (VAL,TEST)
    hold=df[df.timestamp.between(lo,hi,inclusive='left')].sort_values(['asset_id','timestamp'])
    train=df[df.timestamp.between(START,VAL,inclusive='left')]
    bundles=[]; results=[]
    for asset in ASSETS:
        tr=train[train.asset_id.eq(asset)]; tr=tr[eligible(tr) & ~flag(tr,'is_supply_temp_frozen') & ~flag(tr,'is_return_temp_frozen')]
        h=hold[hold.asset_id.eq(asset)]
        if h.empty or (not a.score_test and len(tr)<100): raise ValueError(f'{asset}: insufficient rows')
        for algorithm, wp, cp in [POLICY[asset]]:
            path=models/f'{asset}.joblib'
            if a.score_test:
                b=joblib.load(path)
                if b['asset_id']!=asset or b['algorithm']!=algorithm or b['features']!=FEATURES or b['score_policy']!='training_percentile_anchors_v2' or b['warning_percentile']!=wp or b['critical_percentile']!=cp or b['sklearn_version']!=sklearn.__version__: raise ValueError(f'Incompatible bundle: {path}')
            else:
                b=fit_asset(tr,asset,algorithm,wp,cp); joblib.dump(b,path,compress=3)
            g=score_readings(h,b); bundles.append(b); results.append(g)
            print(f'{asset} {algorithm}: {b["training_records"]} training rows; {g.anomaly_score.notna().sum()} scored; anchors {b["calibration_anchors"]}',flush=True)
        if not a.score_test:
            score_figure(b,g,out/f'{asset}_score_calibration.png')
            timeline_figure(b,g,out/f'{asset}_anomaly_timeline.png')
    labels=verify(results,Path(a.manifest)); rows=[]
    for i,(b,g) in enumerate(zip(bundles,results)):
        raw=b['sorted_training_scores']; anchors=b['calibration_anchors']; valid=g.anomaly_score.notna(); s=g.anomaly_score[valid]
        r=dict(asset_id=b['asset_id'],algorithm=b['algorithm'],training_records=b['training_records'],warning_percentile=b['warning_percentile'],critical_percentile=b['critical_percentile'],anchor_min=anchors[0],anchor_warning=anchors[1],anchor_critical=anchors[2],anchor_cap=anchors[3],training_alert_rate=float((raw>=anchors[1]).mean()),training_critical_rate=float((raw>=anchors[2]).mean()),records=len(g),scored=int(valid.sum()),unscored=int((~valid).sum()),provisional=int(g.prediction_status.eq('PROVISIONAL').sum()),normal=int(g.classification.eq('NORMAL').sum()),warning=int(g.classification.eq('WARNING').sum()),critical=int(g.classification.eq('CRITICAL').sum()),raw_min=g.raw_anomaly_score.min(),raw_max=g.raw_anomaly_score.max(),score_min=s.min(),score_max=s.max(),native_disagreements=int(((g.native_prediction[valid]==-1)!=(s>=.5)).sum()))
        if labels is not None:
            y=labels[i][valid]; r.update(metrics(y,s>=.5)); r['unscored_actual_anomalies']=int(labels[i][~valid].sum()); r['average_precision']=average_precision_score(y,s) if y.any() else np.nan
            for t in [.30,.40,.50,.60,.75,.90]:
                m=metrics(y,s>=t); r.update({f'{k}_at_{t:.2f}':m[k] for k in ['precision','recall','f1']})
        rows.append(r)
    summary=pd.DataFrame(rows); summary.insert(0,'split',split)
    summary_path=out/'evaluation_summary.csv'
    if a.score_test:
        if not summary_path.exists(): raise FileNotFoundError('Run validation before test evaluation')
        previous=pd.read_csv(summary_path)
        summary_all=pd.concat([previous[previous.split.eq('validation')],summary],ignore_index=True)
    else: summary_all=summary
    summary_all.to_csv(summary_path,index=False)
    if not a.score_test:
        fig,axes=plt.subplots(3,1,figsize=(14,10),sharex=True,constrained_layout=True)
        for ax,b,g in zip(axes,bundles,results):
            timeline(ax,b,g); ax.set_title(f'{b["asset_id"]} · {b["algorithm"]}',loc='left')
        axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%b %d %H:%M',tz=mdates.UTC))
        axes[-1].set_xlabel('UTC')
        fig.suptitle('Detected anomalous readings — validation',fontsize=16)
        fig.savefig(out.parent/'anomalies_detected.png',dpi=160); plt.close(fig)
    print(summary[['asset_id','algorithm','scored','normal','warning','critical']+(['precision','recall','f1'] if labels is not None else [])].to_string(index=False))
    print(f'Saved {split} evaluation summary. '+('No refitting.' if a.score_test else 'Independent test untouched.'))

if __name__=='__main__': main()
