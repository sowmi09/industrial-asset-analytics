"""Generate industrial telemetry charts from cleaned measurements and optional predictions."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

try:
    from .generate_data import ASSET_CONFIG
    from .data_processor import CLEANED_COLUMNS
except ImportError:
    from generate_data import ASSET_CONFIG
    from data_processor import CLEANED_COLUMNS

STATUS_COLORS = {'RUNNING': '#2563eb', 'STANDBY': '#d97706', 'OFF': '#94a3b8'}
QUALITY_COLOR = '#d97706'


def fields(df, column, field):
    return df[column].fillna('').astype(str).str.split('|', regex=False).map(lambda x: field in x)


def quality_mask(df):
    return (df.is_missing_timestamp | df.missing_fields.notna() & df.missing_fields.ne('')
            | df.invalid_fields.notna() & df.invalid_fields.ne('')
            | df.imputed_fields.notna() & df.imputed_fields.ne('')
            | df.is_supply_temp_frozen | df.is_return_temp_frozen)


def shade(ax, times, mask, color, alpha=.20):
    selected = pd.DatetimeIndex(times[mask])
    if not len(selected):
        return
    cuts = np.flatnonzero(np.diff(selected.asi8) != 60 * 10**9) + 1
    for group in np.split(np.arange(len(selected)), cuts):
        ax.axvspan(selected[group[0]], selected[group[-1]] + pd.Timedelta(minutes=1), color=color, alpha=alpha, linewidth=0)


def load_data(path, scored=None):
    df = pd.read_csv(path, dtype={c: "string" for c in ("original_missing_fields", "invalid_fields", "imputed_fields", "missing_fields")})
    missing = set(CLEANED_COLUMNS) - set(df)
    if missing:
        raise ValueError(f'Missing cleaned columns: {sorted(missing)}')
    df.timestamp = pd.to_datetime(df.timestamp, utc=True, errors='raise')
    if df.duplicated(['asset_id','timestamp']).any() or not df.asset_id.isin(ASSET_CONFIG).all():
        raise ValueError('Invalid asset identifiers or duplicate telemetry keys')
    for col in ['is_missing_timestamp','is_supply_temp_frozen','is_return_temp_frozen','is_rolling_power_estimated','is_ml_eligible']:
        vals = df[col].astype(str).str.lower().map({'true': True, 'false': False})
        if vals.isna().any():
            raise ValueError(f'Invalid boolean flag: {col}')
        df[col] = vals
    df['anomaly_score'] = np.nan
    if scored:
        scores = pd.read_csv(scored, usecols=['asset_id','timestamp','anomaly_score'])
        scores.timestamp = pd.to_datetime(scores.timestamp, utc=True, errors='raise')
        if not pd.MultiIndex.from_frame(scores[['asset_id','timestamp']]).isin(pd.MultiIndex.from_frame(df[['asset_id','timestamp']])).all():
            raise ValueError('Scored telemetry contains keys absent from cleaned telemetry')
        scores.anomaly_score = pd.to_numeric(scores.anomaly_score, errors='raise')
        if not scores.anomaly_score.dropna().between(0,1).all():
            raise ValueError('Anomaly scores must lie in [0, 1]')
        df = df.drop(columns='anomaly_score').merge(scores, on=['asset_id','timestamp'], how='left', validate='one_to_one')
        if (df.anomaly_score.notna() & ~df.is_ml_eligible).any():
            raise ValueError('Scores provided for ineligible records')
    return df.sort_values(['asset_id','timestamp'])


def save(fig, path):
    fig.savefig(path, dpi=160, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print('Saved:', path)


def format_time(ax, detail=False):
    if detail:
        locator = mdates.AutoDateLocator(minticks=4, maxticks=7)
        formatter = mdates.DateFormatter('%d %b\n%H:%M')
    else:
        locator = mdates.DayLocator(interval=1)
        formatter = mdates.DateFormatter('%d %b\n%H:%M')
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
    ax.tick_params(axis='x', labelsize=8)
    ax.set_xlabel('Date and time (UTC)' if detail else 'Date (UTC; ticks at 00:00)')


def intervals(g, mask):
    times = pd.DatetimeIndex(g.timestamp[mask])
    if not len(times):
        return []
    cuts = np.flatnonzero(np.diff(times.asi8) != pd.Timedelta(minutes=1).value) + 1
    return [(x[0], x[-1] + pd.Timedelta(minutes=1)) for x in np.split(times, cuts)]


def trend_panels(p, f, g, asset):
    t = g.timestamp
    score = g.anomaly_score
    p.plot(t, g.power_kw, color='#64748b', lw=.85, label='Actual power')
    f.plot(t, g.flow_rate, color='#2563eb', lw=.85, label='Flow')
    if score.notna().any():
        for mask, color, label in [(score.lt(.5), '#15803d', 'NORMAL'),
                                   (score.ge(.5)&score.lt(.75), '#d97706', 'WARNING'),
                                   (score.ge(.75), '#dc2626', 'CRITICAL')]:
            p.scatter(t[mask], g.loc[mask,'power_kw'], s=5, color=color, label=label, zorder=3)
            if label != 'NORMAL':
                for ax in (p,f):
                    shade(ax,t,mask,color,.15)
    p.set_title(asset + (' | ML severity available' if score.notna().any() else ' | Before ML: health not evaluated'),loc='left',fontweight='bold')
    p.set_ylabel('Power (kW)')
    f.set_ylabel(f"Flow ({ASSET_CONFIG[asset]['flow_unit']})")
    for ax in (p,f):
        ax.grid(alpha=.18)
        ax.set_xlim(t.min(),t.max()+pd.Timedelta(minutes=1))
        ax.legend(loc='upper right',fontsize=9,ncol=4)
    p.tick_params(labelbottom=False)
    f.xaxis.set_major_locator(mdates.DayLocator(interval=1))
    f.xaxis.set_major_formatter(mdates.DateFormatter('%d %b\n%H:%M'))
    f.set_xlabel('Date and time (UTC)')


def trends(df,out):
    assets=list(df.asset_id.unique())
    names={'Chiller-01':'chiller','Compressor-02':'compressor','Pump-03':'pump'}
    fig, axes=plt.subplots(len(assets)*2,1,figsize=(24,4.8*len(assets)),squeeze=False,layout='constrained')
    fig.suptitle('Industrial telemetry | Power and flow',fontsize=19,fontweight='bold')
    for i,asset in enumerate(assets):
        g=df[df.asset_id.eq(asset)]
        p,f=axes[2*i:2*i+2,0]
        f.sharex(p)
        trend_panels(p,f,g,asset)
        single,ax=plt.subplots(2,1,figsize=(24,7),sharex=True,layout='constrained')
        trend_panels(*ax,g,asset)
        save(single,out/(names[asset]+'_trend.png'))
    save(fig,out/'trends.png')


def rolling_power(df,out):
    assets=list(df.asset_id.unique())
    fig,axes=plt.subplots(len(assets),1,figsize=(24,3.2*len(assets)),squeeze=False,layout='constrained')
    fig.suptitle('One-hour rolling mean power | Separate from instantaneous trends',fontsize=17,fontweight='bold')
    for ax,asset in zip(axes[:,0],assets):
        g=df[df.asset_id.eq(asset)]
        ax.plot(g.timestamp,g.rolling_power_mean_1h,color='#0f766e',lw=.9)
        ax.set_title(asset,loc='left',fontweight='bold'); ax.set_ylabel('Power (kW)'); ax.grid(alpha=.2)
        format_time(ax)
    save(fig,out/'rolling_power.png')


def quality_lanes(g):
    return [ ('Restored timestamps',g.is_missing_timestamp,'#d97706','s'),
             ('Originally missing sensors',g.original_missing_fields.fillna('').ne(''),'#d97706','s'),
             ('Unresolved sensor values',g.missing_fields.fillna('').ne(''),'#d97706','s'),
             ('Invalid power',fields(g,'invalid_fields','power_kw'),'#dc2626','x'),
             ('Invalid flow / pressure',fields(g,'invalid_fields','flow_rate') | fields(g,'invalid_fields','pressure_bar'),'#dc2626','x'),
             ('Invalid supply temperature',fields(g,'invalid_fields','supply_temp_c'),'#dc2626','x'),
             ('Invalid return temperature',fields(g,'invalid_fields','return_temp_c'),'#dc2626','x'),
             ('Supply temperature freeze',g.is_supply_temp_frozen,'#9333ea','s'),
             ('Return temperature freeze',g.is_return_temp_frozen,'#9333ea','s'),
             ('Imputed sensor values',g.imputed_fields.fillna('').ne(''),'#0f766e','D') ]


def draw_quality(ax,g,annotate=False):
    lanes=quality_lanes(g)
    for y,(label,mask,color,marker) in enumerate(lanes):
        events = intervals(g,mask)
        for start,end in events:
            ax.plot([start,end],[y,y],color=color,lw=5,solid_capstyle='butt')
            ax.scatter([start],[y],color=color,marker=marker,s=32,zorder=4)
        if annotate:
            by_day = {}
            for start,end in events:
                by_day.setdefault(start.normalize(), []).append(start)
            for day, starts in by_day.items():
                label = day.strftime('%d %b') + ' ' + ', '.join(x.strftime('%H:%M') for x in starts)
                if len(starts)>1:
                    label = day.strftime('%d %b') + '\n' + ', '.join(x.strftime('%H:%M') for x in starts)
                ax.annotate(label,(starts[0],y),xytext=(3,5),textcoords='offset points',fontsize=7,color=color)
    ax.set_yticks(range(len(lanes)),[f'{label} ({int(mask.sum())})' for label,mask,_,_ in lanes],fontsize=8)
    ax.set_ylim(len(lanes)-.45,-.55)
    ax.grid(axis='x',alpha=.2)
    ax.grid(axis='y',alpha=.1)
    ax.set_xlim(g.timestamp.min()-pd.Timedelta(hours=2),g.timestamp.max()+pd.Timedelta(hours=6))


def data_quality(df,out):
    assets=list(df.asset_id.unique())
    fig,axes=plt.subplots(len(assets),1,figsize=(18,12),squeeze=False,layout='constrained')
    fig.suptitle('Data-quality events before ML | Counts are affected readings, not equipment failures',fontsize=16,fontweight='bold')
    for ax,asset in zip(axes[:,0],assets):
        g=df[df.asset_id.eq(asset)]
        draw_quality(ax,g,annotate=True)
        ax.set_title(asset,loc='left',fontweight='bold')
        format_time(ax)
    save(fig,out/'data_quality.png')


def anomalies(df,out):
    assets=list(df.asset_id.unique())
    fig,axes=plt.subplots(len(assets)*3,1,figsize=(18,17),squeeze=False,layout='constrained')
    fig.suptitle('Detected anomalies | ML severity, independent quality flags and efficiency',fontsize=17,fontweight='bold')
    for i,asset in enumerate(assets):
        g=df[df.asset_id.eq(asset)]
        a,b,c=axes[3*i:3*i+3,0]; t=g.timestamp; score=g.anomaly_score
        a.axhspan(0,.5,color='#15803d',alpha=.07); a.axhspan(.5,.75,color='#d97706',alpha=.12); a.axhspan(.75,1,color='#dc2626',alpha=.12)
        a.axhline(.5,color='#d97706',ls='--',lw=.9); a.axhline(.75,color='#dc2626',ls='--',lw=.9)
        a.plot(t,score,color='#475569',lw=.65,label='Anomaly score')
        for mask,color,marker,label in [(score.ge(.5)&score.lt(.75),'#d97706','o','WARNING'),(score.ge(.75),'#dc2626','x','CRITICAL')]:
            a.scatter(t[mask],score[mask],color=color,marker=marker,s=8,label=label,zorder=4)
            c.scatter(t[mask],g.loc[mask,'specific_energy'],color=color,marker=marker,s=8,zorder=4)
        a.set_ylim(0,1); a.set_ylabel('Score (0–1)'); a.legend(loc='upper right',fontsize=8,ncol=3)
        a.set_title(asset+' — operational anomaly scores',loc='left',fontweight='bold')
        if not score.notna().any(): a.text(.5,.5,'Not evaluated',transform=a.transAxes,ha='center')
        draw_quality(b,g); b.set_title('Independent data-quality events',loc='left',fontsize=10)
        c.plot(t,g.specific_energy,color='#2563eb',lw=.7)
        shade(c,t,score.ge(.5),'#dc2626',.16)
        c.set_ylabel(f"kW / {ASSET_CONFIG[asset]['flow_unit']}")
        c.set_title('Specific energy — markers indicate ML severity',loc='left',fontsize=10)
        for ax in (a,b,c):
            ax.set_xlim(g.timestamp.min(),g.timestamp.max()+pd.Timedelta(minutes=1)); ax.grid(axis='x',alpha=.16); format_time(ax)
    save(fig,out/'anomalies_detected.png')


def efficiency(df,out):
    assets=list(df.asset_id.unique())
    fig,axes=plt.subplots(len(assets),2,figsize=(16,5*len(assets)),squeeze=False,layout='constrained',gridspec_kw={'width_ratios':[1.8,1]})
    fig.suptitle('Power versus flow | Status colors do not indicate health',fontsize=18,fontweight='bold')
    for (ax,zoom),asset in zip(axes,assets):
        g=df[df.asset_id.eq(asset)]
        valid=np.isfinite(g.power_kw)&np.isfinite(g.flow_rate)
        for col in ['power_kw','flow_rate']:
            valid &= ~fields(g,'invalid_fields',col)&~fields(g,'imputed_fields',col)
        for status,color in STATUS_COLORS.items():
            h=g[valid&g.status.eq(status)]
            marker={'RUNNING':'.','STANDBY':'o','OFF':'s'}[status]
            ax.scatter(h.flow_rate,h.power_kw,s=10,alpha=.45,color=color,marker=marker,label=f'{status}: {len(h):,} readings')
            if status!='RUNNING':
                zoom.scatter(h.flow_rate,h.power_kw,s=35,alpha=.65,color=color,marker=marker,label=f'{status}: {len(h):,}')
        running=g[valid&g.status.eq('RUNNING')]
        r=running.flow_rate.corr(running.power_kw) if len(running)>1 else np.nan
        ax.set_title(f'{asset} | RUNNING Pearson r = {r:.3f}' if np.isfinite(r) else f'{asset} | RUNNING correlation unavailable',fontweight='bold')
        zoom.set_title('STANDBY and OFF — close-up (points can overlap)')
        idle=g[valid&g.status.isin(['STANDBY','OFF'])]
        if not idle.empty:
            xmax=max(1.,float(idle.flow_rate.max())*1.1)
            ymax=max(.1,float(idle.power_kw.max())*1.15)
            zoom.set_xlim(-.05*xmax,xmax)
            zoom.set_ylim(-.08*ymax,ymax)
        for panel in (ax,zoom):
            panel.set_xlabel(f"Flow ({ASSET_CONFIG[asset]['flow_unit']})")
            panel.set_ylabel('Power (kW)')
            panel.grid(alpha=.2)
            panel.legend(fontsize=9,loc='best',markerscale=1.5)
    save(fig,out/'efficiency_correlation.png')


def interactive_charts(df, out):
    """Write offline interactive views without altering telemetry or static charts."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    folder = out / 'interactive'
    folder.mkdir(parents=True, exist_ok=True)
    names = {'Chiller-01': 'chiller', 'Compressor-02': 'compressor', 'Pump-03': 'pump'}
    assets = list(df.asset_id.unique())

    def write(fig, name):
        path = folder / name
        fig.write_html(str(path), include_plotlyjs=True, full_html=True,
                       config={'scrollZoom': True, 'displaylogo': False, 'responsive': True})
        print('Saved:', path)

    def time_controls(fig, row=1):
        fig.update_xaxes(type='date', tickformat='%d %b\n%H:%M',
                         hoverformat='%Y-%m-%d %H:%M:%S UTC')
        fig.update_xaxes(rangeselector=dict(buttons=[
            dict(count=1, label='1h', step='hour', stepmode='backward'),
            dict(count=6, label='6h', step='hour', stepmode='backward'),
            dict(count=1, label='1d', step='day', stepmode='backward'),
            dict(count=7, label='7d', step='day', stepmode='backward'),
            dict(label='All', step='all')]), row=row, col=1)

    for asset in assets:
        g = df[df.asset_id.eq(asset)]
        unit = ASSET_CONFIG[asset]['flow_unit']
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=.10,
                            subplot_titles=('Actual power', 'Flow rate'))
        # Explicit UTC strings keep the displayed clock consistent in every browser.
        ts = g.timestamp.dt.strftime('%Y-%m-%d %H:%M:%S').tolist()
        custom = g[['status','power_kw','flow_rate']].fillna('Unavailable').to_numpy()
        hover = ('%{x|%Y-%m-%d %H:%M:%S} UTC<br>Status: %{customdata[0]}'
                 '<br>Power: %{customdata[1]} kW<br>Flow: %{customdata[2]} '+unit+'<extra></extra>')
        for row, col, name, color in [(1,'power_kw','Actual power','#475569'),(2,'flow_rate','Flow','#2563eb')]:
            fig.add_trace(go.Scattergl(x=ts,y=g[col].tolist(),mode='lines',name=name,
                          line=dict(color=color,width=1.4),connectgaps=False,
                          customdata=custom,hovertemplate=hover),row=row,col=1)
        for label,mask,color in [('NORMAL',g.anomaly_score.lt(.5),'#15803d'),
                                 ('WARNING',g.anomaly_score.ge(.5)&g.anomaly_score.lt(.75),'#d97706'),
                                 ('CRITICAL',g.anomaly_score.ge(.75),'#dc2626')]:
            if mask.any():
                fig.add_trace(go.Scattergl(x=np.array(ts)[mask.to_numpy()].tolist(),y=g.loc[mask,'power_kw'].tolist(),
                              mode='markers',name=label,marker=dict(color=color,size=5),
                              customdata=g.loc[mask,['anomaly_score']].to_numpy(),
                              hovertemplate='%{x|%Y-%m-%d %H:%M:%S} UTC<br>Power: %{y:.3f} kW<br>Score: %{customdata[0]:.3f}<extra>'+label+'</extra>'),row=1,col=1)
        fig.update_yaxes(title_text='Power (kW)',row=1,col=1)
        fig.update_yaxes(title_text=f'Flow ({unit})',row=2,col=1)
        fig.update_xaxes(title_text='Date and time (UTC)',row=2,col=1)
        time_controls(fig)
        fig.update_layout(title=f'{asset} | Power and flow — '+('ML scores available' if g.anomaly_score.notna().any() else 'Before ML'),
                          template='plotly_white',height=720,hovermode='x unified',dragmode='zoom',
                          margin=dict(t=125,b=65),legend=dict(orientation='h',y=-.15))
        write(fig,names[asset]+'_trend.html')

    fig = make_subplots(rows=len(assets),cols=2,subplot_titles=[f'{a} — {s}' for a in assets for s in ('All statuses','STANDBY / OFF close-up')])
    for row,asset in enumerate(assets,1):
        g=df[df.asset_id.eq(asset)]
        unit=ASSET_CONFIG[asset]['flow_unit']
        valid=np.isfinite(g.power_kw)&np.isfinite(g.flow_rate)
        for col in ('power_kw','flow_rate'):
            valid &= ~fields(g,'invalid_fields',col)&~fields(g,'imputed_fields',col)
        for status,color in STATUS_COLORS.items():
            h=g[valid&g.status.eq(status)].groupby(['flow_rate','power_kw'],as_index=False).size()
            for column in ([1] if status=='RUNNING' else [1,2]):
                fig.add_trace(go.Scattergl(x=h.flow_rate.tolist(),y=h.power_kw.tolist(),mode='markers',
                              name=f'{asset} {status} ({int(h["size"].sum()):,} readings)',
                              marker=dict(color=color,size=6,opacity=.6),showlegend=column==1,
                              customdata=h[['size']].to_numpy(),
                              hovertemplate='Flow: %{x:.4f} '+unit+'<br>Power: %{y:.4f} kW<br>Identical readings: %{customdata[0]}<extra>'+status+'</extra>'),row=row,col=column)
        for column in (1,2):
            fig.update_xaxes(title_text=f'Flow ({unit})',row=row,col=column)
            fig.update_yaxes(title_text='Power (kW)',row=row,col=column)
    fig.update_layout(title='Power vs flow | Status, not equipment health | Identical coordinates grouped',template='plotly_white',height=460*len(assets),dragmode='zoom')
    write(fig,'efficiency_correlation.html')

    fig=make_subplots(rows=len(assets),cols=1,shared_xaxes=True,subplot_titles=assets,vertical_spacing=.10)
    for row,asset in enumerate(assets,1):
        g=df[df.asset_id.eq(asset)]
        lanes=quality_lanes(g)
        for label,mask,color,_ in lanes:
            for start,end in intervals(g,mask):
                count=int((end-start)/pd.Timedelta(minutes=1))
                # End is exclusive: each row represents a one-minute interval.
                details=f'{label}<br>Start: {start:%Y-%m-%d %H:%M} UTC<br>Last reading: {end-pd.Timedelta(minutes=1):%Y-%m-%d %H:%M} UTC<br>Affected readings: {count}'
                fig.add_trace(go.Scatter(x=[start.strftime('%Y-%m-%d %H:%M:%S'),end.strftime('%Y-%m-%d %H:%M:%S')],y=[label,label],
                              mode='lines+markers',line=dict(color=color,width=5),marker=dict(size=7),
                              text=[details,details],hovertemplate='%{text}<extra></extra>',showlegend=False),row=row,col=1)
        fig.update_yaxes(categoryorder='array',categoryarray=[x[0] for x in reversed(lanes)],row=row,col=1)
        fig.update_xaxes(range=[g.timestamp.min().strftime('%Y-%m-%d %H:%M:%S'),g.timestamp.max().strftime('%Y-%m-%d %H:%M:%S')],row=row,col=1)
    time_controls(fig)
    fig.update_layout(title='Data-quality events | Cleaned telemetry flags only',template='plotly_white',height=420*len(assets),margin=dict(l=210,t=125),dragmode='zoom')
    write(fig,'data_quality.html')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',default='data/cleaned_telemetry.csv')
    parser.add_argument('--scored',default=None,help='Optional actual scored telemetry CSV')
    parser.add_argument('--output-dir',default='outputs')
    args=parser.parse_args()
    df=load_data(args.input,args.scored)
    out=Path(args.output_dir)
    out.mkdir(parents=True,exist_ok=True)
    trends(df,out)
    rolling_power(df,out)
    efficiency(df,out)
    data_quality(df,out)
    interactive_charts(df,out)
    if args.scored and df.anomaly_score.notna().any():
        anomalies(df,out)
    else:
        print('ML anomaly chart deferred: no actual model scores supplied.')
    print('Records plotted:',len(df))
    print('Model-scored records:',int(df.anomaly_score.notna().sum()))


if __name__=='__main__':
    main()
