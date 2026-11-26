"""Analyze feature redundancy on reliable historical training telemetry."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

FEATURES = ['power_kw', 'flow_rate', 'supply_temp_c', 'return_temp_c',
            'pressure_bar', 'delta_t', 'specific_energy', 'load_ratio']
ASSETS = ['Chiller-01', 'Compressor-02', 'Pump-03']
START = pd.Timestamp('2026-10-01', tz='UTC')
END = pd.Timestamp('2026-10-09', tz='UTC')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', default='data/cleaned_telemetry.csv')
    parser.add_argument('--output', default='outputs/feature_analysis')
    args = parser.parse_args()
    df = pd.read_csv(args.input)
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True, errors='raise')
    if df.duplicated(['asset_id', 'timestamp']).any():
        raise ValueError('Duplicate asset timestamps')
    for col in FEATURES:
        df[col] = pd.to_numeric(df[col], errors='raise')
    mask = (df.timestamp.between(START, END, inclusive='left')
            & df.status.eq('RUNNING')
            & df.is_ml_eligible.astype(str).str.lower().eq('true')
            & np.isfinite(df[FEATURES]).all(axis=1))
    training = df.loc[mask]
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    pairs = []
    for asset in ASSETS:
        x = training.loc[training.asset_id.eq(asset), FEATURES]
        if len(x) < 2:
            raise ValueError(f'{asset}: insufficient eligible training data')
        print(f'\n{asset}: {len(x)} eligible training readings')
        constant = x.columns[x.nunique() <= 1].tolist()
        if constant:
            print('Constant features (undefined correlations):', constant)
        fig, axes = plt.subplots(1, 2, figsize=(19, 9), layout='constrained')
        matrices = {method: x.corr(method=method) for method in ('pearson', 'spearman')}
        for ax, (method, corr) in zip(axes, matrices.items()):
            corr.to_csv(out / f'{asset}_{method}.csv')
            im = ax.imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1)
            ax.set_xticks(range(len(FEATURES)), FEATURES, rotation=45, ha='right')
            ax.set_yticks(range(len(FEATURES)), FEATURES)
            ax.set_title(method.title() + ' correlation')
            for i in range(len(FEATURES)):
                for j in range(len(FEATURES)):
                    v = corr.iloc[i, j]
                    ax.text(j, i, f'{v:.2f}' if np.isfinite(v) else 'N/A',
                            ha='center', va='center', color='white' if abs(v) > .65 else 'black')
        fig.colorbar(im, ax=axes, shrink=.65, label='Correlation coefficient')
        fig.suptitle(f'{asset} | Eligible RUNNING training data | 01–08 Oct 2026 UTC | n={len(x):,}')
        fig.savefig(out / f'{asset}_correlation.png', dpi=150)
        plt.close(fig)
        for i, a in enumerate(FEATURES):
            for b in FEATURES[i+1:]:
                p, s = (matrices[m].loc[a,b] for m in ('pearson','spearman'))
                strong = (abs(p) >= .9 or abs(s) >= .9)
                pairs.append(dict(asset_id=asset, feature_a=a, feature_b=b, pearson=p,
                                  spearman=s, strong_association=strong, training_records=len(x)))
                if strong:
                    print(f'  {a} / {b}: Pearson={p:.4f}, Spearman={s:.4f}')
    pd.DataFrame(pairs).to_csv(out / 'feature_correlations.csv', index=False)
    print(f'\nSaved feature analysis to {out}')
    print('Strong association is a review flag, not an automatic feature removal rule.')


if __name__ == '__main__':
    main()
