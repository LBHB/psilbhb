import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

from psilbhb.stim.wav_set import MCWavFileSet, FgBgSet, VowelSet

pd.set_option('display.width',160)

if os.path.exists('h:/sounds'):
    soundpath = 'H:/sounds/vowels/v6/'
else:
    soundpath = '/auto/data/sounds/vowels/v6'


params = VowelSet.default_values()
params.update(dict(sound_path=soundpath,
                   target_set= 	['01_AE106+03_AE106', '02_AE151+04_AE151', '13_EH106+15_EH106', '14_EH151+16_EH151'],
                   non_target_set=['17_OO106+19_OO106', '18_OO151+20_OO151'],
                   catch_set=['05_AW106+11_EE106'],
                   switch_channels=False, repeat_count=2,
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
          d['s1_name'], d['s2_name'], d['response_condition'], d['trial_is_repeat'])

raise ValueError('stopping')

# plot waveforms from an example trial
for trial_idx in range(1, 5):
    w = fb.trial_waveform(trial_idx)
    d = fb.trial_parameters(trial_idx)
    print(trial_idx, w.shape)
    if d['this_fg_level'] == 0:
        fg_scaleby = 0
    else:
        fg_scaleby = 10 ** ((d['this_fg_level'] - fb.FgSet.level) / 20)
    if d['this_bg_level'] == 0:
        bg_scaleby = 0
    else:
        bg_scaleby = 10 ** ((d['this_bg_level'] - fb.BgSet.level) / 20)

    if d['response_condition']==-1:
        wb = fb.FgSet.waveform(d['bg_i'])*bg_scaleby
    else:
        wb = fb.BgSet.waveform(d['bg_i'])*bg_scaleby
    wf = fb.FgSet.waveform(d['fg_i'])*fg_scaleby
    print(trial_idx, w.shape, wb.shape, wf.shape)

    f, ax = plt.subplots(2,1, sharex='col', sharey='col')
    t=np.arange(w.shape[1])/fb.FgSet.fs
    ax[0].plot(t, w[0, :])
    if d['bg_channel'] == 0:
        ax[0].plot(t, wb[:, 0])
    ax[0].set_title('channel 1')
    if w.shape[1]>1:
        ax[1].plot(t, w[1,:], label='f+b')
    if d['bg_channel']==1:
        ax[1].plot(t,wb[:,0], label='b')
    ax[1].legend()
    ax[1].set_title('channel 2')

    ax[0].set_title(f"trial {trial_idx} fgc={d['fg_channel']} bgc={d['bg_channel']} fgdB={d['this_fg_level']}")

    plt.tight_layout()
