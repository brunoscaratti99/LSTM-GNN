"""Inspect stored data/checkpoints without training or changing their artifacts."""
from pathlib import Path
import json
import hashlib
import numpy as np
import pandas as pd
import torch
import xarray as xr

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
RUNS = ROOT/'Experiments/run_experiment/07_09_2026'


def stats(x):
    x=np.asarray(x)
    return dict(min=float(x.min()),median=float(np.median(x)),max=float(x.max()),mean=float(x.mean()))


def main():
    runs=sorted(RUNS.iterdir())
    state=json.loads((runs[-1]/'inference_state.json').read_text())
    names=state['stations']
    coords=np.asarray([state['station_coordinates'][n] for n in names])
    edge_index=np.asarray(state['edge_index'])
    radians=np.radians(coords)
    delta_coords=radians[edge_index[0]]-radians[edge_index[1]]
    hav=np.sin(delta_coords[:,0]/2)**2+np.cos(radians[edge_index[0],0])*np.cos(radians[edge_index[1],0])*np.sin(delta_coords[:,1]/2)**2
    distances=2*6371*np.arcsin(np.sqrt(hav))
    sel=dict(latitude=xr.DataArray(coords[:,0],dims='station',coords={'station':names}),
             longitude=xr.DataArray(coords[:,1],dims='station',coords={'station':names}))
    daily=xr.open_zarr(ROOT/'Datasets/processed/daily/precipitation_daily.zarr')
    yda=daily.tp.sel(time=slice('2000-01-01','2025-12-31')).sel(**sel,method='nearest').transpose('time','station').load()
    y=yda.values.astype(float)
    dates=pd.DatetimeIndex(yda.time.values)
    train_end=int(len(y)*.6)
    val_end=train_end+int(len(y)*.2)
    train=y[:train_end]
    pd.DataFrame({'station':names,'train_mean_mm':train.mean(0)}).to_csv(OUT/'train_station_climatology.csv',index=False)
    pd.DataFrame([{'month':month,'station':name,'train_mean_mm':value}
                  for month in range(1,13)
                  for name,value in zip(names,train[dates[:train_end].month==month].mean(0))]).to_csv(OUT/'train_station_month_climatology.csv',index=False)
    data={'knn_distances_km':stats(distances),'directed_edges':len(distances)}
    for name,sl in [('train',slice(0,train_end)),('val',slice(train_end,val_end)),('test',slice(val_end,None))]:
        z=y[sl]
        data[name]=dict(shape=list(z.shape),start=str(dates[sl][0].date()),end=str(dates[sl][-1].date()),
                        nan=int(np.isnan(z).sum()),mean=float(z.mean()),std=float(z.std()),
                        dry_le_1_fraction=float((z<=1).mean()),heavy_gt_35_fraction=float((z>35).mean()))
    bins=np.searchsorted([.2999305725097656,15.209197998046875],train,side='left')
    w=np.array([.41375964879989624,.5172774791717529,2.068962812423706])[bins]
    data['global_weighted_constant_mm']=float((train*w).sum()/w.sum())
    data['identical_target_pairs']=[(names[i],names[j]) for i in range(len(names)) for j in range(i+1,len(names)) if np.array_equal(y[:,i],y[:,j])]
    # ERA5 hourly accumulation valid at 00:00 belongs to the preceding UTC day.
    # Correct daily sum = saved 00..23 sum + next midnight - current midnight.
    raw=xr.open_zarr(ROOT/'Datasets/processed/era5_precipitation_80-26.zarr')
    midnight=raw.tp.sel(time=slice('2023-12-31','2024-12-31'),step=np.timedelta64(6,'h'))
    midnight=midnight.sel(time=midnight.time.dt.hour==18).sel(**sel,method='nearest').transpose('time','station').values.astype(float)*1000
    delta=midnight[1:]-midnight[:-1]
    saved=yda.sel(time=slice('2024-01-01','2024-12-31')).values
    assert delta.shape==saved.shape
    data['utc_day_correction_2024']=dict(shape=list(delta.shape),mae=float(np.abs(delta).mean()),
        p95_abs=float(np.quantile(np.abs(delta),.95)),max_abs=float(np.abs(delta).max()),
        fraction_abs_gt_1=float((np.abs(delta)>1).mean()),
        changed_class_gt_35=int(((saved>35)!=((saved+delta)>35)).sum()))
    (OUT/'data_inspection.json').write_text(json.dumps(data,indent=2),encoding='utf-8')

    summaries={}
    checkpoints=[]
    for run in runs:
        if not (run/'hist.pt').exists():
            continue
        hist=torch.load(run/'hist.pt',map_location='cpu',weights_only=False)
        cfg=json.loads((run/'config.json').read_text())
        ckpt=torch.load(run/'model_state_dict.pt',map_location='cpu',weights_only=True)
        checkpoints.append(ckpt)
        best=int(np.argmin(np.asarray(hist['val_mse'])[cfg['warm_up']:]))+cfg['warm_up']
        minloss=int(np.argmin(hist['val_loss']))
        z=ckpt['cell_0.a_logits']
        mask=ckpt['cell_0.topology_mask']
        prior=ckpt.get('cell_0.edge_weight_prior')
        prior_used=bool(ckpt.get('cell_0.uses_edge_weight_prior',False))
        if prior_used:
            edge_mask=ckpt['cell_0.edge_mask']
            off=prior*torch.exp(z)*edge_mask+torch.sigmoid(z)*(mask-edge_mask)
        else:
            off=torch.sigmoid(z)*mask
        off=(off+off.T)*.5
        a=torch.eye(len(names))+off
        dn=a.sum(1).rsqrt()
        an=dn[:,None]*a*dn[None,:]
        s=dict(epochs=len(hist['val_loss']),selected_epoch=best+1,
               selected_val_loss=hist['val_loss'][best],min_loss_epoch=minloss+1,min_val_loss=hist['val_loss'][minloss],
               selected_train_prediction_loss=hist['train_prediction_loss'][best],
               selected_val_prediction_loss=hist['val_prediction_loss'][best],
               final_train_prediction_loss=hist['train_prediction_loss'][-1],final_val_prediction_loss=hist['val_prediction_loss'][-1],
               prior_used=prior_used,raw_edge_weights=stats(off[mask.bool()].numpy()),
               normalized_diagonal=stats(an.diag().numpy()),normalized_offdiag_row_sum=stats((an.sum(1)-an.diag()).numpy()))
        if prior_used:
            s['prior_edge_weights']=stats(prior[mask.bool()].numpy())
            initial=torch.eye(len(names))+prior
            di=initial.sum(1).rsqrt()
            initial_norm=di[:,None]*initial*di[None,:]
            s['initial_normalized_diagonal']=stats(initial_norm.diag().numpy())
        aux=pd.read_csv(run/'test_node_standard_deviation_predictions_by_lead_day.csv')
        ay=aux.actual_node_std_mm.to_numpy()
        ap=aux.predicted_node_std_mm.to_numpy()
        s['auxiliary_std']=dict(actual_mean=float(ay.mean()),predicted_mean=float(ap.mean()),r2=float(1-((ay-ap)**2).sum()/((ay-ay.mean())**2).sum()))
        summaries[run.name]=s
    summaries['first_two_checkpoints_equal']=all(torch.equal(checkpoints[0][k],checkpoints[1][k]) for k in checkpoints[0])
    (OUT/'checkpoint_inspection.json').write_text(json.dumps(summaries,indent=2),encoding='utf-8')
    sources=[ROOT/'src/run_experiment.py',ROOT/'src/Models/model.py',ROOT/'src/Training/Training_Routines.py',ROOT/'src/Training/experiment_runner.py',ROOT/'src/Data/temporal_dataset.py',ROOT/'src/Data/feature_extraction.py',ROOT/'src/Graph/graph_related_utils.py']
    (OUT/'source_sha256.json').write_text(json.dumps({str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},indent=2),encoding='utf-8')
    print(json.dumps(data,indent=2))
    print(json.dumps(summaries,indent=2))


if __name__=='__main__':
    main()
