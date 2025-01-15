import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

from psilbhb.stim.wav_set import MCWavFileSet, FgBgSet, VowelSet

pd.set_option('display.width',160)

if os.path.exists('h:/sounds'):
    soundpath_fg = 'h:/sounds/vocalizations/v4'
    # soundpath_bg = 'h:/sounds/backgrounds/v3'
    soundpath_bg = 'h:/sounds/Categories/temp_bgs'

    # Satya testing
    # soundpath_prb = 'h:/sounds/Categories/chimeric_voc'
    #soundpath_prb_bg = 'h:/sounds/Categories/temp_probes'
    #soundpath_prb_fg = 'h:/sounds/Categories/temp_probes'
    prb_bg_range = [2, 7, 10, 11, 14, 15]
    # Jonah testing
    soundpath_prb_bg = ''
    prb_bg_range=[]
else:
    soundpath_fg = '/auto/data/sounds/vocalizations/v4'
    soundpath_bg = '/auto/data/sounds/backgrounds/v3'
    soundpath_prb_bg = '/auto/data/sounds/Categories/chimeric_voc'

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
# Update probe trials indices based on the get_stim_list()
params.update(dict(fg_path=soundpath_fg, fg_range=[6, 7],
                   bg_path=soundpath_bg, bg_range=[3,4,5],
                   prb_bg_path=soundpath_prb_bg, prb_bg_range=prb_bg_range,
                   prb_f=2, fg_choice_trials=2,
                 fg_switch_channels=True, contra_n=1, ipsi_n=1, diotic_n=1,
                 combinations='all', migrate_fraction=0.0, fg_delay=0.5, duration=2.0,
                 fg_level=[55, 63], bg_level=[55], random_seed=4234))

# params.update(dict(fg_path=soundpath_fg, bg_path=soundpath_bg,
#                    prb_path=soundpath_prb, fg_range=[1,2], bg_range=[0],
#                    prb_range=[69, 70, 71, 73, 74, 99, 100, 101, 103, 104, 189, 190, 191, 193, 194],
#                  fg_switch_channels=True, contra_n=1, ipsi_n=1, diotic_n=1,
#                  fg_choice_trials=2,
#                  combinations='all', migrate_fraction=0.0, fg_delay=0.5, duration=2.0,
#                  fg_level=[0, 55], bg_level=[0, 55], random_seed=4234))

fb = FgBgSet(2, **params)
fb.update()  # not necessary but illustrative of back-end processing

# simulated_performance = [0, 0, 2, 2, 2, 1, 2, 2, 1, 2, 2, 1, 1, 2, 0,
#                          2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 1, 1, 2, 0,
#                          1, 1, 0, 1, 2, 0, 1, 2]
simulated_performance = [2] * 50

N=50
fg_chan = np.zeros(N)
bg_chan = np.zeros(N)
for trial_idx in range(len(simulated_performance)):
    d = fb.trial_parameters(trial_idx+1)
    fb.score_response(simulated_performance[trial_idx], repeat_incorrect=True, trial_idx=trial_idx+1)
    d = fb.trial_parameters(trial_idx+1)
    print(d['trial_idx'], d['wav_set_idx'], d['current_full_rep'],
          d['fg_name'], d['fg_channel'],
          d['bg_name'], d['bg_channel'], d['this_snr'],
          d['trial_cat'], d['response_condition'], d['current_full_rep'], d['trial_is_repeat'],
          simulated_performance[trial_idx])
    fg_chan[trial_idx] = d['fg_channel']
    bg_chan[trial_idx] = d['bg_channel']

raise ValueError('stopping')
print(fg_chan.mean(), bg_chan.mean())


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
