"""Read-only analysis of saved forecasts; writes audit tables beside this script."""
from pathlib import Path
import json
import hashlib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
RUNS = ROOT / 'Experiments/run_experiment/07_09_2026'


def metrics(y, p):
    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    error = p - y
    denom = np.sum((y - y.mean()) ** 2)
    return dict(n=len(y), rmse=float(np.sqrt(np.mean(error**2))),
                mae=float(np.mean(np.abs(error))), bias=float(error.mean()),
                r2=float(1-np.sum(error**2)/denom) if denom else None,
                actual_mean=float(y.mean()), predicted_mean=float(p.mean()),
                actual_std=float(y.std()), predicted_std=float(p.std()),
                predicted_min=float(p.min()), predicted_max=float(p.max()),
                corr=float(np.corrcoef(y, p)[0, 1]) if p.std() > 0 else None)


def categorical(y, p, threshold):
    actual, forecast = y > threshold, p > threshold
    tp = int(np.sum(actual & forecast))
    fp = int(np.sum(~actual & forecast))
    fn = int(np.sum(actual & ~forecast))
    return dict(threshold=threshold, tp=tp, fp=fp, fn=fn,
                actual_frequency=float(actual.mean()), forecast_frequency=float(forecast.mean()),
                recall=tp/(tp+fn) if tp+fn else None,
                precision=tp/(tp+fp) if tp+fp else None,
                csi=tp/(tp+fp+fn) if tp+fp+fn else None)


def main():
    climatology = pd.read_csv(OUT / 'train_station_climatology.csv')
    monthly = pd.read_csv(OUT / 'train_station_month_climatology.csv')
    print('Baseline columns:', climatology.columns.tolist(), monthly.columns.tolist())
    summaries, rows, detection, spatial = {}, [], [], []
    reference_targets = None
    for run in sorted(RUNS.iterdir()):
        path = run / 'test_predictions_by_lead_day.csv'
        if not path.exists():
            continue
        df = pd.read_csv(path, usecols=['sample','lead_day','target_time','station','actual_mm','predicted_mm'])
        targets = df[['sample','lead_day','target_time','station','actual_mm']]
        if reference_targets is None:
            reference_targets = targets
        else:
            assert targets.equals(reference_targets), 'Runs must share identical dated targets.'
        y, p = df.actual_mm.to_numpy(), df.predicted_mm.to_numpy()
        summaries[run.name] = dict(all=metrics(y,p), heavy=metrics(y[y>35],p[y>35]),
                                   negative_fraction=float((p<0).mean()),
                                   target_start=df.target_time.min(), target_end=df.target_time.max(),
                                   csv_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        for lead, group in [('all', df), *list(df.groupby('lead_day'))]:
            y, p = group.actual_mm.to_numpy(), group.predicted_mm.to_numpy()
            for label, mask in [('all',np.ones(len(y),dtype=bool)),('dry_le_1',y<=1),('wet_gt_1',y>1),('heavy_gt_35',y>35)]:
                rows.append(dict(run=run.name, lead=lead, subset=label, **metrics(y[mask],p[mask])))
            for threshold in [1., 10., 35.]:
                detection.append(dict(run=run.name,lead=lead,**categorical(y,p,threshold)))
        for lead, group in df.groupby('lead_day'):
            yg = group.pivot(index='sample',columns='station',values='actual_mm').to_numpy()
            pg = group.pivot(index='sample',columns='station',values='predicted_mm').to_numpy()
            ys, ps = yg.std(axis=1), pg.std(axis=1)
            for subset, mask in [('all',np.ones(len(yg),dtype=bool)),('any_station_gt_35',(yg>35).any(axis=1))]:
                spatial.append(dict(run=run.name,lead=lead,subset=subset,n=int(mask.sum()),
                                    actual_spatial_std=float(ys[mask].mean()),
                                    predicted_spatial_std=float(ps[mask].mean()),
                                    spread_retention=float(ps[mask].mean()/ys[mask].mean()),
                                    spread_temporal_corr=float(np.corrcoef(ys[mask],ps[mask])[0,1])))
    pd.DataFrame(rows).to_csv(OUT/'forecast_metrics.csv',index=False)
    pd.DataFrame(detection).to_csv(OUT/'event_detection.csv',index=False)
    pd.DataFrame(spatial).to_csv(OUT/'spatial_dispersion.csv',index=False)
    (OUT/'output_summary.json').write_text(json.dumps(summaries,indent=2),encoding='utf-8')
    print(json.dumps(summaries,indent=2))

    # Persistence uses the last observed date before the forecast origin's D+1.
    # First test origin is excluded when its previous observation is absent.
    df['target_time'] = pd.to_datetime(df.target_time)
    observations = df[['target_time','station','actual_mm']].drop_duplicates(['target_time','station'])
    forecasts = df.copy()
    forecasts['origin'] = forecasts.target_time - pd.to_timedelta(forecasts.lead_day,unit='D')
    persistence = observations.rename(columns={'target_time':'origin','actual_mm':'persistence_mm'})
    forecasts = forecasts.merge(persistence,on=['origin','station'],how='left')
    forecasts = forecasts.merge(climatology[['station','train_mean_mm']],on='station',how='left')
    forecasts['month'] = forecasts.target_time.dt.month
    month_col = [c for c in monthly if c not in ('station','month')][0]
    forecasts = forecasts.merge(monthly.rename(columns={month_col:'monthly_mean_mm'}),on=['station','month'],how='left')
    baselines=[]
    for lead,group in [('all',forecasts),*list(forecasts.groupby('lead_day'))]:
        y=group.actual_mm.to_numpy()
        for name,col in [('persistence','persistence_mm'),('train_station_mean','train_mean_mm'),('train_station_month_mean','monthly_mean_mm')]:
            p=group[col].to_numpy()
            valid = np.isfinite(p)
            for subset,mask in [('all',valid),('heavy_gt_35',(y>35)&valid)]:
                baselines.append(dict(model=name,lead=lead,subset=subset,**metrics(y[mask],p[mask])))
    pd.DataFrame(baselines).to_csv(OUT/'baseline_metrics.csv',index=False)
    print('BASELINES',pd.DataFrame(baselines).query("lead == 'all'").to_json(orient='records',indent=2))
    print('LEAD METRICS',pd.DataFrame(rows).query("subset == 'all'")[["run","lead","rmse","r2","bias"]].to_string(index=False))
    print('EVENT METRICS',pd.DataFrame(detection).query("lead == 'all'").to_string(index=False))


if __name__ == '__main__':
    main()
