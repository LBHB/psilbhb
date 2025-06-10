import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

from psilbhb.stim.wav_set import BinauralTone, RandomTone, BandpassNoise
from nems0.analysis.gammatone.gtgram import gtgram, chunked_gtgram

pd.set_option('display.width',160)

params = RandomTone.default_values()
params.update(primary_channel=1, reference_frequency=2000, switch_channels=False, random_seed=4234)

bt = RandomTone(**params)
bt.update()  # not necessary but illustrative of back-end processing

N = 50
for trial_idx in range(N):
    d = bt.trial_parameters(trial_idx+1)
    print(d['trial_idx'], d['wav_set_idx'], d['current_full_rep'],
          d['this_reference_frequency'], d['this_probe_frequency'],
          d['this_snr'], d['current_full_rep'])

# plot waveforms from an example trial
f, ax = plt.subplots(3, 3, figsize=(6, 6), sharex=True, sharey=True)
ax = ax.flatten()
for trial_idx, a in enumerate(ax):
    w = bt.trial_waveform(trial_idx=trial_idx+1)
    d = bt.trial_parameters(trial_idx=trial_idx+1)
    a.plot(w[0, :])
    a.plot(w[1, :]+1)
    a.set_title(d['this_name'], fontsize=8)

    plt.tight_layout()


params = BandpassNoise.default_values()
params.update(primary_channel=0, center=2000, switch_channels=True, random_seed=4234)

bn = BandpassNoise(**params)
bn.update()  # not necessary but illustrative of back-end processing

N = 50
for trial_idx in range(N):
    d = bn.trial_parameters(trial_idx+1)
    print(d['trial_idx'], d['wav_set_idx'], d['current_full_rep'],
          d['this_frequency'], d['this_level'],
          d['current_full_rep'])

# plot waveforms from an example trial
f, ax = plt.subplots(6, 3, figsize=(6, 6), sharex=True, sharey=True)
axsg = ax[1::2,:].flatten()
ax = ax[::2,:].flatten()

rasterfs=200
f_min = 200
f_max = 20000
window_time = 1 / rasterfs
hop_time = 1 / rasterfs
f_nyquist = f_max * 2
padbins = int((window_time - hop_time) / 2 * f_nyquist)
channels=64

for trial_idx, (a,asg) in enumerate(zip(ax,axsg)):
    w = bn.trial_waveform(trial_idx=trial_idx+1)
    d = bn.trial_parameters(trial_idx=trial_idx+1)
    a.plot(w[0, :])
    a.plot(w[1, :]+1)
    a.set_title(d['this_name'], fontsize=8)

    sg = gtgram(np.pad(w.sum(axis=0), [padbins, padbins]), bn.fs, window_time, hop_time, channels, f_min, f_max)
    asg.imshow(sg, origin='lower')
    plt.tight_layout()
