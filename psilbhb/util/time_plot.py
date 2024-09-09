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


rawid1 = 151074
rawid2 = 151083
rawid3 = 151127
rawid4 = 151351

# plot_perform_over_time(rawid1)
# plot_perform_over_time(rawid2)
# plot_perform_over_time(rawid3)
# plot_reaction_over_time(rawid4)
#plot_perform_over_time(rawid4)


rawids = [151394,151396,151466,151467,151511,151545,151574]
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

f,ax = plt.subplots()
ax.plot(d['correct'], color='lightgray', label="raw")

window_len = 12

for i,(s,t) in enumerate(zip(unique_snr,unique_tar)):
    d_ = d.loc[d['this_snr']==s].copy()
    #d_ = d.loc[d['this_target_frequency']==t].copy()
    perf = d_['correct'].mean()
    d_['smooth_correct'] = smooth(d_['correct'].astype(float), window_len=window_len)
    ax.plot(d_['smooth_correct'], label=f"SNR {s}: {perf:.2f}")
    #ax.plot(d_['smooth_correct'], label=f"Tar {t}: {perf:.2f}")
ax.legend(frameon=False)
ax.set_xlabel('Trial')
ax.set_ylabel("Frac. correct")
f.suptitle(parmfile)
