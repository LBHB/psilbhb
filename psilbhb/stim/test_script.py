import matplotlib.pyplot as plt
import numpy as np
import os

from psilbhb.stim.wav_set import MCWavFileSet, FgBgSet, VowelSet

def test_fgbg():
    if os.path.exists('h:/sounds'):
        soundpath_fg = 'h:/sounds/vocalizations/v1'
        soundpath_bg = 'h:/sounds/backgrounds/v1'
    else:
        soundpath_fg = '/auto/data/sounds/vocalizations/v1'
        soundpath_bg = '/auto/data/sounds/backgrounds/v1'

    vv = MCWavFileSet(
        fs=44000, path=soundpath_fg, duration=3, normalization='rms',
        fit_range=slice(0, 1), test_range=None, test_reps=2,
        channel_count=1, level=60)
    bb = MCWavFileSet(
        fs=44000, path=soundpath_bg, duration=4, normalization='rms',
        fit_range=[0, 4], test_range=None, test_reps=2,
        channel_count=1, level=0)

    print(vv.names)

    w = vv.waveform(0)
    print(w.shape)

    fg_snr = -5

    fb = FgBgSet(FgSet=vv, BgSet=bb, fg_switch_channels=True,
                 bg_switch_channels='combinatorial',
                 combinations='all',
                 fg_snr=fg_snr, fg_delay=0.5)
    fb.update()  # not necessary but illustrative of back-end processing

    simulated_performance = [0, 0, 2, 2, 2, 1, 2, 2, 1, 2, 2, 1, 1, 2, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]

    for trial_idx, outcome in enumerate(simulated_performance):
        w = fb.trial_waveform(trial_idx)
        d = fb.trial_parameters(trial_idx)
        fb.score_response(outcome, trial_idx=trial_idx)
        print(d['trial_idx'], d['wav_set_idx'], d['fg_name'], d['fg_channel'], d['bg_name'], d['bg_channel'],
              d['response_condition'], d['current_full_rep'], d['trial_is_repeat'])

    print(f"wav_per_rep: {fb.wav_per_rep}")
    print(f"current full rep: {fb.current_full_rep}")
    print(f"scored trials: {len(fb.trial_outcomes)}")
    print(f"error trials: {sum((fb.trial_outcomes>-1) & (fb.trial_outcomes<2))}")
    print(f"trials remaining this rep: {len(fb.trial_wav_idx)-len(fb.trial_outcomes)}")


    # plot waveforms from an example trial
    trial_idx = 0
    w = fb.trial_waveform(trial_idx)
    wb = fb.BgSet.waveform(fb.bg_index[trial_idx])
    wf = fb.FgSet.waveform(fb.fg_index[trial_idx])

    #for i in range(20):
    #    d = fb.trial_parameters(i)
    #    print(d['response_condition'])

    f, ax = plt.subplots(2,1, sharex='col', sharey='col')
    t=np.arange(w.shape[0])/fb.FgSet.fs
    ax[0].plot(t,w[:,0])
    ax[0].plot(t,wb[:,0])
    ax[0].set_title('channel 1')
    if w.shape[1]>1:
        ax[1].plot(t,w[:,1],label='f+b')
    if wb.shape[1]>1:
        ax[1].plot(t,wb[:,1],label='b')
    ax[1].legend()
    ax[1].set_title('channel 2')
    plt.tight_layout()


def test_vowels_old():
    soundpath_fg = '/auto/data/sounds/vowels/v1'

    #['01_AE_106.wav', '02_AE_151.wav', '03_AE_201.wav',
    # '04_AW_106.wav', '05_AW_151.wav', '06_AW_201.wav',
    # '07_EE_106.wav', '08_EE_151.wav', '09_EE_201.wav',
    # '10_OO_106.wav', '11_OO_151.wav', '12_OO_201.wav',
    # 'x_silence.wav']
    fs=44000
    vv = MCWavFileSet(
        fs=fs, path=soundpath_fg, duration=0.24, normalization='rms',
        fit_range=[0,1,2,6,7,8,6,7,8,3,4,5], test_range=None, test_reps=1,
        channel_count=1, binaural_combinations='all', include_silence=False)
    vv2 = MCWavFileSet(
        fs=fs, path=soundpath_fg, duration=0.24, normalization='rms',
        fit_range=[12,12,12,3,4,5,12,12,12,12,12,12], test_range=None, test_reps=1,
        channel_count=1, binaural_combinations='all', include_silence=False)
    print(vv.names)
    print(len(vv.names))

    w = vv.waveform(0)
    print(w.shape)

    fg_snr = 0

    fb = FgBgSet(FgSet=vv, BgSet=vv2, fg_switch_channels=True,
                 bg_switch_channels='combinatorial',
                 combinations='simple',
                 fg_snr=fg_snr, fg_delay=0.0)
    fb.update()  # not necessary but illustrative of back-end processing

    import sounddevice as sd
    import time
    for wav_set_idx in [1,4,7,10]:
        w=fb.trial_waveform(wav_set_idx=wav_set_idx)
        d=fb.trial_parameters(wav_set_idx=wav_set_idx)
        sd.play(w,fs)
        print(d['trial_idx'], d['wav_set_idx'], d['fg_name'], d['fg_channel'], d['bg_name'], d['bg_channel'])
        time.sleep(0.4)
    simulated_performance = [0, 0, 2, 2, 2, 1, 2, 2, 1, 2, 2, 1, 1, 2, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]

    for trial_idx, outcome in enumerate(simulated_performance):
        w = fb.trial_waveform(trial_idx)
        d = fb.trial_parameters(trial_idx)
        fb.score_response(outcome, trial_idx=trial_idx)

    print(f"wav_per_rep: {fb.wav_per_rep}")
    print(f"current rep: {fb.current_repetition}")
    print(f"scored trials: {len(fb.trial_outcomes)}")
    print(f"error trials: {sum((fb.trial_outcomes > -1) & (fb.trial_outcomes < 2))}")
    print(f"trials remaining this rep: {len(fb.trial_wav_idx) - len(fb.trial_outcomes)}")

    # plot waveforms from an example trial
    trial_idx = 0
    w = fb.trial_waveform(trial_idx)
    wb = fb.BgSet.waveform(fb.bg_index[trial_idx])
    wf = fb.FgSet.waveform(fb.fg_index[trial_idx])

    f, ax = plt.subplots(2, 1, sharex='col', sharey='col')
    t = np.arange(w.shape[0]) / fb.FgSet.fs
    ax[0].plot(t, w[:, 0])
    ax[0].plot(t, wb[:, 0])
    ax[0].set_title('channel 1')
    if w.shape[1] > 1:
        ax[1].plot(t, w[:, 1], label='f+b')
    if wb.shape[1] > 1:
        ax[1].plot(t, wb[:, 1], label='b')
    ax[1].legend()
    ax[1].set_title('channel 2')
    plt.tight_layout()

    d = fb.trial_parameters(trial_idx)

    #plt.show()

sound_path = '/auto/data/sounds/vowels/v2'

v = VowelSet(sound_path=sound_path, switch_channels=True,
             repeat_count=3)
v.update(1)
print(v.trial_wav_idx)

simulated_performance = [0, 0, 2, 2, 2, 1, 2, 2, 1, 2, 2, 1, 1, 2, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]

for trial_idx, outcome in enumerate(simulated_performance):
    w = v.trial_waveform(trial_idx)
    d = v.trial_parameters(trial_idx)
    v.score_response(outcome, trial_idx=trial_idx)

import sounddevice as sd
import time
fs=v.wavset.fs
for wav_set_idx in [1,7,10, 15, 20]:
    w=v.trial_waveform(wav_set_idx=wav_set_idx)
    d=v.trial_parameters(wav_set_idx=wav_set_idx)
    sd.play(w.T,fs)
    print(d['trial_idx'], d['wav_set_idx'], d['s1_name'], d['s2_name'])
    time.sleep(1)
