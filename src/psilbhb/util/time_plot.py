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
from psilbhb.util.plots import smooth, timecourse_plot

rawids = [151467,151511,151545,151574,
          151586, 151623, 151651, 151689, 151691,
          151739]
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

    d_ = d.loc[d.score > 0].copy()
    v = np.roll(d_['score'].values, 1)
    v[0] = 2
    d_['prev_score'] = v
    d_ = d_.loc[d_['prev_score'] == 2]

    dlist.append(d_)

L = [len(d_) for d_ in dlist]
L = np.cumsum(L)
d = pd.concat(dlist, ignore_index=True)

d['this_target_frequency'] = d['this_target_frequency'].astype(int)
d['this_distractor_frequency'] = d['this_distractor_frequency'].astype(int)

f,ax = plt.subplots(3,1,figsize=(10,6))
ax[0].plot(d['correct'], color='lightgray', label="raw")

window_len = 17

timecourse_plot(d, column='this_snr', label='SNR', ax=ax[0],
                window_len=window_len)
[ax[0].axvline(l, linestyle='--', lw=0.5, color='g') for l in L]

timecourse_plot(d, column='this_target_frequency', label='Tar', ax=ax[1],
                window_len=window_len)
[ax[1].axvline(l, linestyle='--', lw=0.5, color='g') for l in L]

timecourse_plot(d, column='this_distractor_frequency', label='Dis', ax=ax[2],
                window_len=window_len)
[ax[2].axvline(l, linestyle='--', lw=0.5, color='g') for l in L]

ax[2].set_xlabel('Trial')

f.suptitle(parmfile)
