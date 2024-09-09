import datetime
import json
import logging

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

log = logging.getLogger(__name__)

c = celldb()

def smooth(x,window_len=11,window='hanning', axis=-1):
    """smooth the data using a window with requested size.

    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.

    input:
        x: the input signal
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal

    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)

    see also:

    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter

    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """

    if x.shape[axis] < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")

    if window_len<3:
        return x

    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError("Window is one of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")
    if window_len & 0x1:
        w1 = int((window_len+1)/2)
        w2 = int((window_len+1)/2)
    else:
        w1 = int(window_len/2)+1
        w2 = int(window_len/2)

    #print(len(s))

    if window == 'flat': #moving average
        w=np.ones(window_len,'d')
    else:
        w=eval('np.'+window+'(window_len)')

    y = convolve1d(x, w/w.sum(), axis=axis, mode='reflect')

    return y

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
            # only include trials where prev trial was correct
            d_ = d_.loc[d_['prev_score']==2]
            print(f"Keeping {d_.shape[0]}/{df_trial.shape[0]} valid trials (not repeat or early np)")
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
    elif runclass in ['NTD']:
        perfsum = d_.groupby(['snr'])[['correct']].mean()
        perfsum = perfsum.unstack(-1)
        perfcount = d_.groupby(['snr'])[['correct']].count()
        perfcount = perfcount.unstack(-1)

        dbias = d_.groupby(['response', 'snr'])['correct'].mean()
        dbias = dbias.unstack(-1)
        width = 12
    elif runclass in ['AMF']:
        perfsum = d_.groupby(['this_snr', 'this_distractor_frequency'])[['correct']].mean()
        perfsum = perfsum.unstack(-1)
        perfcount = d_.groupby(['this_snr', 'this_distractor_frequency'])[['correct']].count()
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
        width = 8

    f, ax = plt.subplots(1,3, figsize=(width, 5))
    perfsum.plot.bar(ax=ax[0], legend=False)
    ax[0].axhline(y=0.5, color='b', linestyle=':')
    ax[0].axhline(y=0.5, color='b', linestyle=':')
    ax[0].set_ylabel('Frac. correct')
    perfcount.plot.bar(ax=ax[1], legend=True)
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
        log.info(f"saving to {figfile}")
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
            log.info(f"error on rawid {rawid}")
    return rawids
