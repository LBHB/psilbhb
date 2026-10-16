import matplotlib.pyplot as plt
import numpy as np
import os

from psilbhb.stim.wav_set import MCWavFileSet, FgBgSet
from psilbhb.stim.bignat import BigNaturalSequenceFactory

if os.path.exists('h:/sounds'):
    soundpath = 'h:/sounds/BigNat/v2'
else:
    soundpath = '/auto/data/sounds/BigNat/v2'

bb1 = BigNaturalSequenceFactory(fs=44000, path=soundpath, level=60, duration=18,
                               normalization="fixed", norm_fixed_scale=25,
                               fit_range=slice(356,456,None), fit_reps=1,
                               test_range=slice(3,5,None), test_reps=5,
                               channel_config=2.1)
bb2 = BigNaturalSequenceFactory(fs=44000, path=soundpath, level=60, duration=18,
                               normalization="fixed", norm_fixed_scale=25,
                               fit_range=slice(356,456,None), fit_reps=1,
                               test_range=slice(3,5,None), test_reps=5,
                               channel_config=2.2)

bb = BigNaturalSequenceFactory(fs=44000, path=soundpath, level=60, duration=18,
                               normalization="fixed", norm_fixed_scale=25,
                               fit_range=slice(20,25,None), fit_reps=1,
                               test_range=slice(3,5,None), test_reps=5,
                               channel_config=1.1)

[[w1.filename.stem,w2.filename.stem] for w1,w2 in zip(bb1.wav_files,bb2.wav_files)]

[w.filename.stem for w in bb.wav_files]

