import matplotlib.pyplot as plt
import numpy as np

from wav_set import BigNaturalSequenceSet, FgBgSet

soundpath = '/auto/data/sounds/BigNat/v2'

bnt = BigNaturalSequenceSet(
    fs=44000, path=soundpath, duration=4, normalization='rms',
    fit_range=slice(6, 16), test_range=slice(0,2), test_reps=2,
    channel_count=2, binaural_combinations='single_offset')

print(bnt.names)

soundpath = '/auto/data/sounds/vocalizations/v1'

vv = BigNaturalSequenceSet(
    fs=44000, path=soundpath, duration=3, normalization='rms',
    fit_range=slice(0, 4), test_range=None, test_reps=2,
    channel_count=1)

print(vv.names)

w = vv.waveform(0)
print(w.shape)

fb = FgBgSet(FgSet=vv, BgSet=bnt, fg_switch_channels=True,
             fg_delay=0.5)
fb.update()
trial_index=3
w = fb.get_trial_waveform(trial_index)
wb = fb.BgSet.waveform(fb.bg_index[trial_index])

f, ax = plt.subplots(2,1)
t=np.arange(w.shape[0])/fb.FgSet.fs
ax[0].plot(t,w[:,0])
ax[0].plot(t,wb[:,0])
ax[1].plot(t,w[:,1])
ax[1].plot(t,wb[:,1])
