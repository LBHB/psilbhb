from functools import cached_property
import itertools
from pathlib import Path

from joblib import Memory
import numpy as np
from scipy import signal
from scipy.io import wavfile

from psiaudio import util, queue
from psiaudio.stim import Waveform, FixedWaveform, ToneFactory, \
    WavFileFactory, WavSequenceFactory, wavs_from_path, load_wav
from psi import get_config


import logging
log = logging.getLogger(__name__)


memory = Memory(get_config('CACHE_ROOT'))


def remove_clicks(w, max_threshold=10, verbose=False):
    w_clean = w

    # log compress everything > 67% of max
    crossover = 0.67 * max_threshold
    ii = (w>crossover)
    w_clean[ii] = crossover + np.log(w_clean[ii]-crossover+1);
    jj = (w<-crossover)
    w_clean[jj] = -crossover - np.log(-w_clean[jj]-crossover+1);

    if verbose:
       print(f'bins compressed down: {ii.sum()} up: {jj.sum()} max {np.abs(w).max():.2f}-->{np.abs(w_clean).max():.2f}')

    return w_clean


@memory.cache
def load_wav(fs, filename, level, calibration, normalization='pe', norm_fixed_scale=1,
             force_duration=None):
    '''
    Load wav file, scale, and resample

    Parameters
    ----------
    fs : float
        Desired sampling rate for wav file. If wav file sampling rate is
        different, it will be resampled to the correct sampling rate using a
        FFT-based resampling algorithm.
    filename : {str, Path}
        Path to wav file
    level : float
        Level to present wav files at. If normalization is `'pe'`, level will
        be in units of peSPL (assuming calibration is in units of SPL). If
        normalization is in `'rms'`, level will be dB SPL RMS.
    calibration : instance of Calibration
        Used to scale waveform to appropriate peSPL. If not provided,
        waveform is not scaled.
    normalization : {'pe', 'rms', 'fixed'}
        Method for rescaling waveform. If `'pe'`, rescales to peak-equivalent
        so the max value of the waveform matches the target level. If `'rms'`,
        rescales so that the RMS value of the waveform matches the target
        level. If 'fixed', scale by a fixed value (norm_fixed_scale)
    norm_fixed_scale : float
        if normalization=='fixed', multiply the wavform by this value.
    force_duration : {None, float}
        if not None, truncate or zero-pad waveform to force_duration sec
    '''
    #log.warning('Loading wav file %r', locals())
    file_fs, waveform = wavfile.read(filename, mmap=True)
    # Rescale to range -1.0 to 1.0
    if waveform.dtype != np.float32:
        ii = np.iinfo(waveform.dtype)
        waveform = waveform.astype(np.float32)
        waveform = (waveform - ii.min) / (ii.max - ii.min) * 2 - 1

    if normalization == 'pe':
        waveform = waveform / waveform.max()
    elif normalization == 'rms':
        waveform = waveform / util.rms(waveform)
    elif normalization == 'fixed':
        waveform = waveform * norm_fixed_scale
        waveform = remove_clicks(waveform, max_threshold=15)
    else:
        raise ValueError(f'Unrecognized normalization: {normalization}')

    if calibration is not None:
        sf = calibration.get_sf(1e3, level)
        waveform *= sf

    # hard-coded rails
    waveform[waveform>5]=5
    waveform[waveform<-5]=-5
    #if np.max(np.abs(waveform)) > 5:
    #    raise ValueError('waveform value too large')

    if force_duration is not None:
        final_samples = int(force_duration*file_fs)
        if len(waveform) > final_samples:
            waveform = waveform[:final_samples]
            log.info(f'truncated to {final_samples} samples')
        elif len(waveform) < final_samples:
            waveform = np.concatenate([waveform, np.zeros(final_samples-len(waveform))])
            log.info(f'padded with {final_samples-len(waveform)} samples')

    # Resample if sampling rate does not match
    if fs != file_fs:
        waveform_resampled = util.resample_fft(waveform, file_fs, fs)
        return waveform_resampled

    return waveform


class NatWavFileFactory(FixedWaveform):

    def __init__(self, fs, filename, level=None, calibration=None,
                 normalization='pe', norm_fixed_scale=1, force_duration=None):
        self.fs = fs
        self.filename = filename
        self.level = level
        self.calibration = calibration
        self.normalization = normalization
        self.norm_fixed_scale = norm_fixed_scale
        self.force_duration = force_duration
        self.reset()

    @cached_property
    def waveform(self):
        return load_wav(self.fs, self.filename, self.level, self.calibration,
                        normalization=self.normalization,
                        norm_fixed_scale=self.norm_fixed_scale,
                        force_duration=self.force_duration)


class BigNaturalSequenceFactory(WavSequenceFactory):

    def __init__(self, fs, path, level=None, calibration=None, duration=-1,
                 normalization='pe', norm_fixed_scale=1,
                 fit_range=None, fit_reps=1, test_range=None, test_reps=0,
                 channel_config='1.1', include_silence=True, random_seed=0):
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
        channel_config : str
            X.Y if X > 1, generate sequences for each of the X channels, coordinating fit and test stim. Return
            sequences for stream Y
        random_seed : int
            Random seed to use when initializing queue for random sequence.
        '''
        if duration > 0:
            force_duration = duration
        else:
            force_duration = None

        if type(channel_config) is float:
            channel_config = str(channel_config)
        try:
            self.channel_count = int(channel_config.split(".")[0])
        except:
            self.channel_count = 1
        try:
            self.channel_id = int(channel_config.split(".")[1])
        except:
            self.channel_id = 1

        all_wav = list(sorted(Path(path).glob('*.wav')))
        if self.channel_count:
            silent_f = [f for i,f in enumerate(all_wav) if str(f).endswith('x_silence.wav')]
            fit_wav = all_wav[fit_range] + silent_f
            test_wav = all_wav[test_range] + silent_f

        fit_wav = fit_wav[(self.channel_id-1):] + fit_wav[:(self.channel_id-1)]
        test_wav = test_wav[(self.channel_id-1):] + test_wav[:(self.channel_id-1)]
        if include_silence & (self.channel_count>1):
            test_wav += silent_f

        fit_wav *= fit_reps
        test_wav *= test_reps
        wav = fit_wav + test_wav

        self.fit_names = fit_wav
        self.test_names = test_wav
        self.wav_files = [NatWavFileFactory(fs, filename, level=level,
                                         calibration=calibration,
                                         normalization=normalization, norm_fixed_scale=norm_fixed_scale,
                                         force_duration=force_duration)
                          for filename in wav]

        self.fs = fs
        self.duration = duration
        self.normalization = normalization
        self.norm_fixed_scale = norm_fixed_scale
        self.random_seed = random_seed
        self.include_silence = include_silence

        self.reset()

    def reset(self):
        self.queue = queue.BlockedRandomSignalQueue(self.fs, self.random_seed)
        metadata = [{'filename': w.filename.stem} for w in self.wav_files]
        self.queue.extend(self.wav_files, np.inf, duration=self.duration, metadata=metadata)
