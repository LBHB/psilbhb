from functools import partial
import itertools
from pathlib import Path

import numpy as np
from scipy import signal
from scipy.io import wavfile

from psiaudio import queue

from psiaudio.stim import Waveform, FixedWaveform, ToneFactory, \
    WavFileFactory, WavSequenceFactory, wavs_from_path

import logging
log = logging.getLogger(__name__)



class BigNaturalSequenceFactory(WavSequenceFactory):

    def __init__(self, fs, path, level=None, calibration=None, duration=-1,
                 normalization='pe', norm_fixed_scale=1,
                 fit_range=None, fit_reps=1, test_range=None, test_reps=0):
        '''
        Parameters
        ----------
        fs : float
            Sampling rate of output. If wav file sampling rate is different, it
            will be resampled to the correct sampling rate using a FFT-based
            resampling algorithm.
        path : {str, Path}
            Path to directory containing wav files.
        level : float
            Level to present wav files at (currently peSPL due to how the
            normalization works).
        calibration : instance of Calibration
            Used to scale waveform to appropriate peSPL. If not provided,
            waveform is not scaled.
        duration : {None, float}
            Duration of each wav file. If None, the wav file is loaded at the
            beginning so that durations can be established. For large
            directories, this can slow down the startup time of the program.
            Knowing the exact duration may be important for some downstream
            operations. For example, epoch extraction relative to the
            presentation time of a particular wav file; estimating the overall
            duration of the entire wav sequence, etc.  If you don't have a need
            for these operations and want to speed up loading of wav files, set
            this value to -1 (the default).
        normalization : {'pe', 'rms', 'fixed'}
            Method for rescaling waveform. If `'pe'`, rescales to peak-equivalent
            so the max value of the waveform matches the target level. If `'rms'`,
            rescales so that the RMS value of the waveform matches the target
            level. If 'fixed', scale by a fixed value (norm_fixed_scale)
        norm_fixed_scale : float
            if normalization=='fixed', multiply the wavform by this value.
        '''
        if duration > 0:
            force_duration = duration
        else:
            force_duration = None

        all_wav = list(sorted(Path(path).glob('*.wav')))

        if fit_range is None:
            fit_wav = all_wav
        else:
            fit_wav = all_wav[fit_range]
        fit_wav *= fit_reps
        if test_range is None:
            test_wav = []
        else:
            test_wav = all_wav[test_range] * test_reps
        wav = fit_wav + test_wav

        self.fit_names = fit_wav
        self.test_names = test_wav
        self.wav_files = [WavFileFactory(fs, filename, level=level,
                                         calibration=calibration,
                                         normalization=normalization, norm_fixed_scale=norm_fixed_scale,
                                         force_duration=force_duration)
                          for filename in wav]

        #self.wav_files = wavs_from_path(fs, path, level=level,
        #                                calibration=calibration,
        #                                normalization=normalization, norm_fixed_scale=norm_fixed_scale,
        #                                force_duration=force_duration)
        self.fs = fs
        self.duration = duration
        self.normalization = normalization
        self.norm_fixed_scale = norm_fixed_scale
        self.reset()

    def reset(self):
        self.queue = queue.BlockedRandomSignalQueue(self.fs)
        self.queue.extend(self.wav_files, np.inf, duration=self.duration)

    def next(self, samples):
        return self.queue.pop_buffer(samples)

if __name__ == "__main__":
    wav_path = '/auto/data/sounds/BigNat/v2'
    n = BigNaturalSequenceFactory(10000, wav_path, duration=20, normalization='fixed',
                               norm_fixed_scale=250, fit_range=slice(6,16),
                               test_range=slice(0,2), test_reps=10)
    # should load 4 files with ~2 sec of silence at the end of each
    w = n.next(800000)
    import matplotlib.pyplot as plt
    plt.figure();plt.plot(w[100000:300000])
else:
    pass