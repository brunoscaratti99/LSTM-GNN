"""Create the standalone figure supporting the pipeline audit."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
OUTPUT=ROOT/'audiction/glstm_07_09_2026.png'
summary=json.loads((HERE/'output_summary.json').read_text())
baseline=pd.read_csv(HERE/'baseline_metrics.csv')
events=pd.read_csv(HERE/'event_detection.csv')
spatial=pd.read_csv(HERE/'spatial_dispersion.csv')
names=['glstm_sweep_20260907_'+n for n in ['090603','114458','121609']]
labels=['09:06 / 10:05','11:44','12:16']
colors=['#3a6ea5','#d4842e','#648f68']
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,axs=plt.subplots(2,2,figsize=(12,8.5),layout='constrained')
fig.suptitle('GLSTM · diagnóstico de 07/09/2026\n62 estações · previsões D+1 a D+5 · alvos ERA5',fontsize=16)
ax=axs[0,0]
base=baseline[(baseline.model=='train_station_month_mean') & (baseline.lead=='all') & (baseline.subset=='all')].rmse.iloc[0]
bars=ax.bar(['Climatologia\nestação + mês',*labels],[base,*[summary[n]['all']['rmse'] for n in names]],color=['#93999e',*colors])
ax.bar_label(bars,fmt='%.2f',padding=3)
ax.set(title='Erro em todos os dias',ylabel='RMSE (mm)',ylim=(0,17))
ax=axs[0,1]
freq=[events[(events.run==n)&(events.lead=='all')&(events.threshold==1)].forecast_frequency.iloc[0]*100 for n in names]
bars=ax.bar(['ERA5',*labels],[events.actual_frequency.iloc[0]*100,*freq],color=['#93999e',*colors])
ax.bar_label(bars,fmt='%.2f%%',padding=3)
ax.set(title='Frequência de chuva acima de 1 mm',ylabel='Percentual dos pares de previsão',ylim=(0,115))
ax=axs[1,0]
for n,label,color in zip(names,labels,colors):
    d=spatial[(spatial.run==n)&(spatial.subset=='any_station_gt_35')]
    ax.plot(d.lead,d.spread_retention*100,'o-',color=color,label=label)
ax.axhline(100,color='#777',linestyle=':',label='Dispersão real')
ax.set(title='Dispersão entre estações nos dias de chuva forte',xlabel='Horizonte (dia)',ylabel='Desvio espacial previsto / real (%)',xticks=range(1,6),ylim=(0,110))
ax.legend(fontsize=9,loc='upper right')
ax=axs[1,1]
h=torch.load(ROOT/'Experiments/run_experiment/07_09_2026'/names[1]/'hist.pt',map_location='cpu',weights_only=False)
x=np.arange(1,len(h['train_prediction_loss'])+1)
ax.plot(x,h['train_prediction_loss'],label='Treino',color='#3a6ea5')
ax.plot(x,h['val_prediction_loss'],label='Validação',color='#b5544b')
ax.axvline(13,linestyle=':',color='#333',label='Checkpoint: época 13')
ax.set(title='Execução 11:44: sobreajuste após o checkpoint',xlabel='Época',ylabel='Loss de precipitação (MSE ponderado)')
ax.legend(fontsize=9)
for ax in axs.flat:
    ax.grid(axis='y',alpha=.2)
    ax.set_axisbelow(True)
OUTPUT.parent.mkdir(exist_ok=True)
fig.savefig(OUTPUT,dpi=190)
print(OUTPUT)
