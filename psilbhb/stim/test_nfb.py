import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

from psilbhb.stim.wav_set import MCWavFileSet, FgBgSet, VowelSet

pd.set_option('display.width',160)

if os.path.exists('h:/sounds'):
    soundpath_fg = 'h:/sounds/vocalizations/v4'
    soundpath_bg = 'h:/sounds/backgrounds/v3'
else:
    soundpath_fg = '/auto/data/sounds/vocalizations/v4'
    soundpath_bg = '/auto/data/sounds/backgrounds/v3'

# vv = MCWavFileSet(
#     fs=44000, path=soundpath_fg, duration=3, normalization='rms',
#     fit_range=slice(8, 11), test_range=slice(0,), test_reps=1,
#     channel_count=1, level=60)
# bb = MCWavFileSet(
#     fs=44000, path=soundpath_bg, duration=4, normalization='rms',
#     fit_range=[3, 4, 5, 6, 7, 8, 9, 10], test_range=slice(0), test_reps=2,
#     channel_count=1, level=60)
#
# print(vv.names)
#
# w = vv.waveform(0)
# print(w.shape)
#
# fg_snr = 100

params = FgBgSet.default_values()
params.update(dict(fg_path=soundpath_fg, bg_path=soundpath_bg,
                 fg_range=[1,2], bg_range=[0],
                 fg_switch_channels=True, contra_n=1, ipsi_n=1, diotic_n=1, fg_choice_trials=2,
                 combinations='all', migrate_fraction=0.0, fg_delay=0.5, duration=2.0,
                 fg_level=[0, 55], bg_level=[0, 55], random_seed=4234))

fb = FgBgSet(2, **params)
fb.update()  # not necessary but illustrative of back-end processing

simulated_performance = [0, 0, 2, 2, 2, 1, 2, 2, 1, 2, 2, 1, 1, 2, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]

N=50
fg_chan = np.zeros(N)
bg_chan = np.zeros(N)
for trial_idx in range(N):
    d = fb.trial_parameters(trial_idx)
    print(d['trial_idx'], d['wav_set_idx'], d['fg_name'], d['fg_channel'], d['bg_name'], d['bg_channel'],
          d['response_condition'], d['current_full_rep'], d['trial_is_repeat'])
    fg_chan[trial_idx] = d['fg_channel']
    bg_chan[trial_idx] = d['bg_channel']
print(fg_chan.mean(), bg_chan.mean())


# plot waveforms from an example trial
for trial_idx in range(0,5):
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
    ax[0].plot(t,w[0,:])
    if d['bg_channel'] == 0:
        ax[0].plot(t, wb[:, 0])
    ax[0].set_title('channel 1')
    if w.shape[1]>1:
        ax[1].plot(t, w[1,:], label='f+b')
    if d['bg_channel']==1:
        ax[1].plot(t,wb[:,0],label='b')
    ax[1].legend()
    ax[1].set_title('channel 2')

    ax[0].set_title(f"trial {trial_idx} fgc={d['fg_channel']} bgc={d['bg_channel']} fgdB={d['this_fg_level']}")

    plt.tight_layout()



"""
#for trial_idx, outcome in enumerate(simulated_performance):
#    w = fb.trial_waveform(trial_idx)
#    d = fb.trial_parameters(trial_idx)
#    fb.score_response(outcome, trial_idx=trial_idx)
#    print(d['trial_idx'], d['wav_set_idx'], d['fg_name'], d['fg_channel'], d['bg_name'], d['bg_channel'],
#          d['response_condition'], d['current_full_rep'], d['trial_is_repeat'])

print(f"wav_per_rep: {fb.wav_per_rep}")
print(f"current full rep: {fb.current_full_rep}")
print(f"scored trials: {len(fb.trial_outcomes)}")
print(f"error trials: {sum((fb.trial_outcomes>-1) & (fb.trial_outcomes<2))}")
print(f"trials remaining this rep: {len(fb.trial_wav_idx)-len(fb.trial_outcomes)}")


"""
