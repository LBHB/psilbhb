import matplotlib
matplotlib.use('QtAgg')
import matplotlib.pyplot as plt

import numpy as np
import os
import pandas as pd

from psilbhb.stim.wav_set import BigNat

pd.set_option('display.width', 160)

plt.ion()
#cf1 = 'D:\\cfts\\20241211-135816 Test test right inear_speaker_calibration_chirp'
#cf2 = 'D:\\cfts\\20241211-135742 Test test left inear_speaker_calibration_chirp'

if os.path.exists('e:/sounds'):
    sound_path = 'e:/sounds/BigNat/v2'
else:
    sound_path = '/auto/data/sounds/BigNat/v2'

params = BigNat.default_values()
params.update(primary_channel=0, level=60, sound_path=sound_path,
              fit_binaural='twooffset', test_binaural='oneoffset', include_silence=False,
              fit_range=range(6,51), test_range=range(3,5), test_reps=8, equalize=True,
              binaural_cross='-6dB')
              #, calfile1 = cf1, calfile2=cf2)

bt = BigNat(**params)
bt.update()  # not necessary but illustrative of back-end processing

N = 50
for trial_idx in range(N):
    d = bt.trial_parameters(trial_idx+1)
    print(d['trial_idx'], d['wav_set_idx'], d['current_full_rep'],
          d['s1_name'], d['s2_name'],
          d['trial_is_repeat'])

# plot waveforms from an example trial
f, ax = plt.subplots(3, 3, figsize=(6, 6), sharex=True, sharey=True)
ax = ax.flatten()
for trial_idx, a in enumerate(ax):
    w = bt.trial_waveform(trial_idx=trial_idx+1)
    d = bt.trial_parameters(trial_idx=trial_idx+1)
    a.plot(w[0, :])
    a.plot(w[1, :]+1)
    a.set_title(d['s1_name'], fontsize=8)
    print(w.shape)
    plt.tight_layout()
