import datetime
import json
import numpy as np
import os
import shutil
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from scipy.ndimage import convolve1d

#import settings
from psi import get_config
from psi.util import PSIJsonEncoder
from psilbhb.util.celldb import celldb, readpsievents, readlogs
from psilbhb.util.plots import smooth

def plot_behavior(rawid=None, parmfile=None, save_fig=True):

    if rawid is not None:
        rawdata = c.pd_query(f"SELECT * FROM gDataRaw where id={rawid}")
    else:
        raise ValueError(f"rawid required")

    pendata = c.pd_query(f"SELECT gPenetration.* FROM gPenetration INNER JOIN gCellMaster ON gPenetration.id=gCellMaster.penid WHERE gCellMaster.id={rawdata.loc[0, 'masterid']}")
    name = pendata.loc[0, 'animal'].lower()
    animal = pendata.loc[0, 'penname'][:3]
    parmfile = rawdata.loc[0, 'parmfile']
    runclass = rawdata.loc[0, 'runclass']

    df_trial, df_event = readlogs(rawid=rawid, c=c)

    d_=df_trial.loc[df_trial.score>0].copy()
    if runclass=='NFB':
        d_['config'] = 'contra'
        d_.loc[(d_['bg_channel']==d_['fg_channel']), 'config']='ipsi'
        d_.loc[(d_['bg_channel']==-1), 'config']='diotic'
        perfsum=d_.groupby(['snr','config'])[['correct']].mean()
        perfsum=perfsum.unstack(-1)
        perfcount=d_.groupby(['snr','config'])[['correct']].count()
        perfcount=perfcount.unstack(-1)

        dbias = d_.groupby(['response','snr'])['correct'].mean()
        dbias = dbias.unstack(-1)
        width=12
    else:
        d_.loc[(d_['s1idx'] == 0) & (d_['s1_name'].astype(str) == 'nan'), 's1_name'] = "AE.828.1920.2500.106.wav"
        d_.loc[(d_['s2idx'] == 0) & (d_['s2_name'].astype(str) == 'nan'), 's2_name'] = "AE.828.1920.2500.106.wav"
        perfsum = d_.groupby(['s1_name'])[['correct']].mean()
        perfcount = d_.groupby(['s1_name'])[['correct']].count()
        dbias = d_.groupby(['response','snr'])['correct'].mean()
        dbias = dbias.unstack(-1)
        width=8


    f,ax = plt.subplots(1,3, figsize=(width,4))
    perfsum.plot.bar(ax=ax[0], legend=True)
    ax[0].axhline(y=0.5,color='b',linestyle=':')
    ax[0].set_ylabel('Frac. correct')
    perfcount.plot.bar(ax=ax[1])
    ax[1].set_ylabel('n trials')
    dbias.plot.bar(ax=ax[2])
    ax[2].set_title('Bias')
    ax[2].set_ylabel('Frac. correct')
    f.suptitle(parmfile)

    plt.tight_layout()


    #plt.show()

    if save_fig:
        remote_root=get_config('REMOTE_ROOT')
        year = pendata.loc[0,'pendate'].split("-")[0]
        savepath = remote_root / 'web' / 'behaviorcharts' / name / year
        savepath.mkdir(parents=True, exist_ok=True)
        figfile = savepath / f"{parmfile}.jpg"
        print(f"saving to {figfile}")
        f.savefig(figfile)

    return df_trial


rawids = [151394,151396,151466,151467,151511,151545,151574,151586]
c = celldb()

dlist = []
for rawid in rawids:
    rawdata = c.pd_query(f"SELECT * FROM gDataRaw where id={rawid}")
    pendata = c.pd_query(f"SELECT gPenetration.* FROM gPenetration INNER JOIN gCellMaster ON gPenetration.id=gCellMaster.penid WHERE gCellMaster.id={rawdata.loc[0, 'masterid']}")
    name = pendata.loc[0, 'animal'].lower()
    animal = pendata.loc[0, 'penname'][:3]
    parmfile = rawdata.loc[0, 'parmfile']
    runclass = rawdata.loc[0, 'runclass']

    d, df_event = readlogs(rawid=rawid, c=c)
    dlist.append(d)
d=pd.concat(dlist, ignore_index=True)

d = d.loc[(d.score > 0) & (d['trial_is_repeat']==0)].reset_index()

unique_snr = d['this_snr'].unique()
unique_tar = d['this_target_frequency'].unique()

f,ax = plt.subplots(2,1,figsize=(8,5))
ax[0].plot(d['correct'], color='lightgray', label="raw")

window_len = 12

for i,s in enumerate(unique_snr):
    d_ = d.loc[d['this_snr']==s].copy()
    perf = d_['correct'].mean()
    d_['smooth_correct'] = smooth(d_['correct'].astype(float), window_len=window_len)
    ax[0].plot(d_['smooth_correct'], label=f"SNR {s}: {perf:.2f}")
ax[0].axhline(0.5, linestyle='--', color='k')
ax[0].legend(frameon=False)

for i,t in enumerate(unique_tar):
    d_ = d.loc[d['this_target_frequency']==t].copy()
    perf = d_['correct'].mean()
    d_['smooth_correct'] = smooth(d_['correct'].astype(float), window_len=window_len)
    ax[1].plot(d_['smooth_correct'], label=f"Tar {t}: {perf:.2f}")
ax[1].axhline(0.5, linestyle='--', color='k')
ax[1].legend(frameon=False)

ax[1].set_xlabel('Trial')
ax[0].set_ylabel("Frac. correct")
ax[1].set_ylabel("Frac. correct")
f.suptitle(parmfile)
