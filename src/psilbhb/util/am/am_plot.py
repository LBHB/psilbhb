import datetime
import json
import numpy as np
import os
import shutil
from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from scipy.ndimage import convolve1d

#import settings
from psi import get_config
from psi.util import PSIJsonEncoder
from psilbhb.util.celldb import celldb, readpsievents, readlogs
from plotsII import smooth, timecourse_plot,timecourse_bar

SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 12
USE_THIS_SIZE = BIGGER_SIZE

plt.rc('font', size=USE_THIS_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=USE_THIS_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=USE_THIS_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=USE_THIS_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=USE_THIS_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
plt.rc('figure', titlesize=USE_THIS_SIZE)  # fontsize of the figure title
mpl.rcParams['font.family'] = 'Arial'
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

plt.ion()
rawids = [#152223, 152224,
          152230, 152232,
          152248,
          152251, 152252,
          152265, 152266,
          152268, 152269,
          152272, 152273,
          152278, 152279,
        152286, 152287,
        152292, 152293,
        152303, 152304,
        152308, 152309,
        152328, 152330,
        152380, 152386,
        152504, 152509,
        152536, 152539
]
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
    d['rawid'] = rawid
    d['session'] = rawdata['cellid'].values[0]

    if False:
        d_ = d.copy()
        v = np.roll(d_['score'].values, 1)
        v[0] = 2
        d_['prev_score'] = v
        d_ = d_.loc[d_['prev_score'] == 2]
        d_ = d_.loc[d.score > 0].copy()
    else:
        d_ = d.loc[d.score > 0].copy()
        v = np.roll(d_['score'].values, 1)
        v[0] = 2
        d_['prev_score'] = v
        d_ = d_.loc[d_['prev_score'] == 2]

    dlist.append(d_)

L = [len(d_) for d_ in dlist]
L = np.cumsum(L)
d = pd.concat(dlist, ignore_index=True)

d['this_distractor_offset'] = d['this_distractor_offset'].astype(float)
#d['this_distractor_frequency'] = d['this_distractor_frequency'].astype(int)
d['target_frequency'] = d['target_frequency']

f,ax = plt.subplots(3,1,figsize=(10,6))
ax[0].plot(d['correct'], color='lightgray', label="raw")

window_len = 17

timecourse_plot(d, column='this_snr', label='SNR', ax=ax[0],
                window_len=window_len)
[ax[0].axvline(l, linestyle='--', lw=0.5, color='g') for l in L]

timecourse_plot(d, column='target_frequency', label='Dis', ax=ax[1],
                window_len=window_len)
[ax[1].axvline(l, linestyle='--', lw=0.5, color='g') for l in L]

ax[1].set_xlabel('Trial')

timecourse_bar(d, column='this_distractor_offset', label='Oct', ax=ax[2],
                window_len=window_len)

f.suptitle(parmfile)

f,ax=plt.subplots()
sess_avg=d.groupby(['this_distractor_offset', 'session'])['correct'].mean()
sess_avg=sess_avg.reset_index()

a = sess_avg.groupby('session').mean()
bad_easy = a['correct']<0.5
bad_sess = a.loc[bad_easy].index.to_list()
#bad_easy = ((sess_avg['this_distractor_offset']==0) & (sess_avg['correct']<0.5))
#bad_easy = ((sess_avg['this_distractor_offset'].abs()==0.6) & (sess_avg['correct']<0.25))
#bad_sess = sess_avg.loc[bad_easy,'session'].to_list()
sess_avg = sess_avg.loc[~sess_avg['session'].isin(bad_sess)]

# remove am-only trials
easy_sess = sess_avg.loc[sess_avg['this_distractor_offset'].abs()==0].copy()
sess_avg = sess_avg.loc[sess_avg['this_distractor_offset'].abs()>0]

sess_avg.plot.scatter(x='this_distractor_offset', y='correct', color='lightgray', ax=ax, label='single day')
sess_avg.groupby('this_distractor_offset').mean(numeric_only=True).reset_index().plot(x='this_distractor_offset', y='correct', color='k', ax=ax)
#ax.plot([0],easy_sess['correct'].mean(),'s', lw=2, label='AM-only')
#d.groupby('this_distractor_offset').mean(numeric_only=True).reset_index().plot(x='this_distractor_offset', y='correct', ax=ax)
ax.set_title(f"Ichy rawids {np.min(rawids)}-{np.max(rawids)}")
ax.set_xlabel('Distractor offset (oct)')
ax.set_ylabel('Mean fraction correct')
ax.legend()

plt.tight_layout()
if os.path.exists('/home/svd/Documents/onedrive/proposals/r01_BinauralFusion/'):
    f.savefig(f'/home/svd/Documents/onedrive/proposals/r01_BinauralFusion/figures/afm_behavior.pdf')
else:
    print("Skipping figure save")
