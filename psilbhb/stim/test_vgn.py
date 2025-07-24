import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

from psilbhb.stim.wav_set import MCWavFileSet, FgBgSet, VowelSet

pd.set_option('display.width',160)

if os.path.exists('h:/sounds'):
    soundpath = 'H:/sounds/vowels/v5/'
else:
    soundpath = '/auto/data/sounds/vowels/v5'


params = VowelSet.default_values()
params.update(dict(sound_path=soundpath,
                   target_set=['01_AE_106+01_AE_106', '02_AE_151+02_AE_151', '01_AE_106+02_AE_151'],
                   non_target_set=['04_AW_106+04_AW_106', '05_AW_151+05_AW_151', '04_AW_106+05_AW_151'],
                   catch_set=['04_AW_106+10_OO_106', '05_AW_151+11_OO_151', '05_AW_151+10_OO_106', '04_AW_106+11_OO_151'], mono_pairs=True,
                   switch_channels=True, repeat_count=2, n_response=1,
                   random_seed=4234))

v = VowelSet(1, **params)
v.update()  # not necessary but illustrative of back-end processing

simulated_performance = [0, 0, 2, 2, 2, 1, 2, 2, 1, 2, 2, 1, 1, 2, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]

N=26
for trial_idx in range(N):
    d = v.trial_parameters(trial_idx+1)
    v.score_response(simulated_performance[trial_idx], repeat_incorrect=True, trial_idx=trial_idx+1)
    d = v.trial_parameters(trial_idx+1)
    print('score', v.trial_outcomes[trial_idx], 'trial_idx', d['trial_idx'],
          'wav_set_idx', d['wav_set_idx'], 'rep', d['current_full_rep'],
          d['this_name'], d['response_condition'], d['trial_is_repeat'])

# plot waveforms from example trials
f, ax = plt.subplots(3, 3, figsize=(8, 8), sharex=True, sharey=True)
ax = ax.flatten()
for trial_idx, a in enumerate(ax):
    w = v.trial_waveform(trial_idx=trial_idx+1)
    d = v.trial_parameters(trial_idx=trial_idx+1)
    a.plot(w[0, :])
    a.plot(w[1, :]+1)
    a.set_title(d['this_name'], fontsize=8)
    print(w.shape)
    plt.tight_layout()
