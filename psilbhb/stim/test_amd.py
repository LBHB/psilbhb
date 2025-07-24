import matplotlib
#matplotlib.use('QtAgg')
import matplotlib.pyplot as plt

import numpy as np
import os
import pandas as pd

from psilbhb.stim.wav_set import BinauralTone, BinauralAM, AMDetect

pd.set_option('display.width', 160)

params = AMDetect.default_values()
params.update(primary_channel=0, switch_channels=False, random_seed=4234,
              go_frequency=[400], nogo_frequency=[400], distractor_offset=[-0.5, 0, 0.5],
              target_level=[55], distractor_level=[55], go_multiplier=1,
              go_depth=[0, 5, 10, 15, 20, 25], nogo_depth=[60], include_mono=True)

bt = AMDetect(**params)
bt.update()  # not necessary but illustrative of back-end processing

N = 50
for trial_idx in range(N):
    d = bt.trial_parameters(trial_idx+1)
    print(d['trial_idx'], d['wav_set_idx'], d['current_full_rep'],
          d['this_name'], d['trial_is_repeat'])

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
