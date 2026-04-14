import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

from psilbhb.stim.wav_set import BinauralToneFusion

pd.set_option('display.width',160)

params = BinauralToneFusion.default_values()
params.update(primary_channel=0, switch_channels=True, random_seed=4234, fs=44000,
              reference_center=[1000,2000],
              probe_octaves=[-1, -0.3, -0.1, 0.1, 0.3, 1],
              probe_alone_octaves=[-1, -0.6, -0.3, -0.1, 0, 0.1, 0.3, 0.6, 1],
              probe_level=[0],
              probe_delay=[0,1],
              include_mono=True)

bt = BinauralToneFusion(**params)
bt.update()  # not necessary but illustrative of back-end processing

N = 50
for trial_idx in range(N):
    d = bt.trial_parameters(trial_idx+1)
    print(d['trial_idx'], d['wav_set_idx'], d['current_full_rep'],
          d['this_reference_frequency'], d['this_probe_frequency'],
          d['this_snr'],
          d['current_full_rep'])

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
