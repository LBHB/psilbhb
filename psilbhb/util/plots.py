import datetime
import json
import numpy as np
import os
import shutil
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

#import settings
from psi import get_config
from psi.util import PSIJsonEncoder
from psilbhb.util.celldb import celldb, readpsievents, readlogs

c = celldb()

def plot_behavior(rawid=None, parmfile=None, save_fig=True):

    if rawid is None:
        raise ValueError(f"rawid required")
    df_list = []
    if type(rawid) is list:
        rawid_list = rawid
    else:
        rawid_list = [rawid]

    for rawid in rawid_list:
        rawdata = c.pd_query(f"SELECT * FROM gDataRaw where id={rawid}")

        pendata = c.pd_query(f"SELECT gPenetration.* FROM gPenetration INNER JOIN gCellMaster ON gPenetration.id=gCellMaster.penid WHERE gCellMaster.id={rawdata.loc[0, 'masterid']}")
        name = pendata.loc[0, 'animal'].lower()
        animal = pendata.loc[0, 'penname'][:3]
        parmfile = rawdata.loc[0, 'parmfile']
        runclass = rawdata.loc[0, 'runclass']

        df_trial, df_event = readlogs(rawid=rawid, c=c)

        # throw out invalid trials-- early NP or previous trial was error
        if df_trial.score.dtype == 'O':
            # convert from HIT/MISS to
            df_trial['score_str']=df_trial['score']
            df_trial['score']=0
            # 2: hit = 'HIT'     0: miss = 'MISS' 3: correct_reject = 'CR'    1: false_alarm = 'FA'

            df_trial.loc[df_trial['score_str']=='FA','score']=1
            df_trial.loc[df_trial['score_str']=='HIT','score']=2
            df_trial.loc[df_trial['score_str']=='CR','score']=3
            df_trial['correct'] = df_trial['score']>=2

            d_=df_trial.copy()
            v = np.roll(d_['score'].values,1)
            v[0]=2
            d_['prev_score']=v
            d_ = d_.loc[d_['prev_score']>=2]

        else:
            d_=df_trial.loc[df_trial.score>0].copy()
            v = np.roll(d_['score'].values,1)
            v[0]=2
            d_['prev_score']=v
            d_ = d_.loc[d_['prev_score']==2]

        df_list.append(d_)

    d_ = pd.concat(df_list, ignore_index=True)
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
    elif runclass == 'NTD':
        perfsum = d_.groupby(['snr'])[['correct']].mean()
        perfsum = perfsum.unstack(-1)
        perfcount = d_.groupby(['snr'])[['correct']].count()
        perfcount = perfcount.unstack(-1)

        dbias = d_.groupby(['response', 'snr'])['correct'].mean()
        dbias = dbias.unstack(-1)
        width = 12
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


def fix_old_plots(sitemask="SQD"):

    sql = f"SELECT * FROM gDataRaw WHERE not(bad) AND isnull(trials) AND cellid like '{sitemask}%'"
    #sql = f"SELECT * FROM gDataRaw WHERE not(bad) AND cellid like '{sitemask}%'"
    rawdata = c.pd_query(sql)
    rawids = []
    for i, r in rawdata.iterrows():
        rawid = r['id']
        print(i, rawid, r['resppath'])
        try:
            c.refresh_rawdata(r['id'])
            rawids.append(rawid)
            plot_behavior(rawid=rawid)
        except:
            print(f"error on rawid {rawid}")
    return rawids
