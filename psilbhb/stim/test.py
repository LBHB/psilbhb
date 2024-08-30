import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

from psilbhb.stim.wav_set import BinauralTone

pd.set_option('display.width',160)

params = BinauralTone.default_values()
params.update(primary_channel=0, switch_channels=False, random_seed=4234)

bt = BinauralTone(**params)
bt.update()  # not necessary but illustrative of back-end processing


N=50
for trial_idx in range(N):
    d = bt.trial_parameters(trial_idx+1)
    print(d['trial_idx'], d['wav_set_idx'], d['current_full_rep'],
          d['this_reference_frequency'], d['this_probe_frequency'],
          d['this_snr'],
          d['current_full_rep'], d['trial_is_repeat'])

# plot waveforms from an example trial
f,ax = plt.subplots(5, 5, figsize=(8, 8), sharex=True, sharey=True)
ax=ax.flatten()
for trial_idx, a in enumerate(ax):
    w = bt.trial_waveform(trial_idx+1)
    d = bt.trial_parameters(trial_idx+1)
    a.plot(w[0,:])
    a.plot(w[1,:]+1)
    a.set_title(d['this_name'], fontsize=8)

    plt.tight_layout()
