
from functools import partial
import itertools
from pathlib import Path

import numpy as np
from scipy import signal
from scipy.io import wavfile

from psiaudio import queue

from psiaudio.stim import Waveform, FixedWaveform, ToneFactory, \
    WavFileFactory, WavSequenceFactory, wavs_from_path, load_wav

import logging
log = logging.getLogger(__name__)



class WaveformSet():

    def __init__(self, fs=40000, level=65, calibration=None, channel_count=1):
        self.level = level
        self.fs = fs
        self.calibration = calibration
        self.channel_count = channel_count

    @property
    def max_index(self):
        return 0

    @property
    def names(self):
        return []

    def waveform(self, idx):
        return np.array([])


class WavFileSet(WaveformSet):

    def __init__(self, filenames, normalization='pe',
                 norm_fixed_scale=1, force_duration=None, random_seed=0, **kwargs):
        """
        :param filenames: list filenames (single channel) or list of lists
           (each file in inner list fed to a different channel)
        :param normalization:
        :param norm_fixed_scale:
        :param force_duration:
        :param random_seed:
        :param kwargs: passthrough parameters to WaveformSet
        """
        self.filenames = filenames
        self.normalization = normalization
        self.norm_fixed_scale = norm_fixed_scale
        self.force_duration = force_duration
        self.random_seed = random_seed

        super().__init__(**kwargs)

    def waveform(self, idx):
        files = self.filenames[idx]
        if type(files) is str:
            files = [files]

        w = [load_wav(self.fs, f, self.level, self.calibration,
                      normalization=self.normalization,
                      norm_fixed_scale=self.norm_fixed_scale,
                      force_duration=self.force_duration)
             for f in files]
        return np.stack(w, axis=1)

    @property
    def names(self):
        l = []
        for file in self.filenames:
            if type(file) is str:
                file = [file]

            s = "+".join([_f.name for _f in file])
            l.append(s)

        return l

    @property
    def max_index(self):
        return len(self.filenames)

    @property
    def index_sorted(self):
        _rng = np.random.RandomState(self.random_seed)
        i = np.arange(self.max_index)
        _rng.shuffle(i)
        return i

    @property
    def wave_list_raw(self):
        return [self.wav_files[i].filename for i in range(self.max_index) ]

    @property
    def wave_list_sorted(self):
        return [self.wav_files[i].filename.stem for i in self.index_sorted ]

    def get_by_index(self, idx):
        return self.wav_files[self.index_sorted[idx]].waveform()


class BigNaturalSequenceSet(WavFileSet):

    def __init__(self, path, duration=-1, include_silence=True,
                 fit_range=None, fit_reps=1, test_range=None, test_reps=0,
                 channel_count=1, channel_offset=1, binaural_combinations='single_offset',
                 **kwargs):
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
        channel_count : int
            Currently only really 1 or 2 (binaural)
        binaural_combinations : str {'single_offset','all}
            default is single_offset
        random_seed : int
            Random seed to use when initializing queue for random sequence.
        '''
        if duration > 0:
            force_duration = duration
        else:
            force_duration = None
        if test_range is None:
            test_range=slice(0)
        if fit_range is None:
            fit_range=slice(0)
            
        all_wav = list(sorted(Path(path).glob('*.wav')))

        if include_silence:
            silent_f = [f for i,f in enumerate(all_wav) if str(f).endswith('x_silence.wav')]
            ff = all_wav[fit_range] + silent_f
            tt = all_wav[test_range] + silent_f
        else:
            ff = all_wav[fit_range]
            tt = all_wav[test_range]

        fit_wav = []
        test_wav = []
        wav = []
        if binaural_combinations == 'single_offset':
            for channel_id in range(channel_count):
                fit_wav.append(ff[(channel_id*channel_offset-1):] + ff[:(channel_id*channel_offset-1)])
                test_wav.append(tt[(channel_id*channel_offset-1):] + tt[:(channel_id*channel_offset-1)])

                fit_wav[-1] *= fit_reps
                test_wav[-1] *= test_reps
                wav.append(fit_wav[-1] + test_wav[-1])
            filenames = []
            for i in range(len(wav[0])):
                filenames.append([])
            for w in wav:
                for i, f in enumerate(w):
                    filenames[i].append(f)

        elif binaural_combinations == 'all':
            fit_wav=[]
            test_wav=[]
            for fi in ff:
                for fj in ff:
                    fit_wav.append([fi,fj])
            for fi in tt:
                for fj in tt:
                    test_wav.append([fi,fj])
            fit_wav *= fit_reps
            test_wav *= test_reps
            filenames = fit_wav+test_wav

        self.channel_offset = channel_offset
        self.fit_names = fit_wav
        self.test_names = test_wav
        self.duration = duration

        super().__init__(filenames, channel_count=channel_count, force_duration=force_duration, **kwargs)


class FgBgSet():
    """
    Target+background class – generic self-initiated trial
        Allows yoked target and background ids
        maybe? Unclear how these should be controlled.

    Methods:
        get_background(idx) – continuous bg, temporally uncoupled from fg
        get_target(idx) –
        can contain background as well,
        waveform plus target location (or indicate no-go)
        trial duration – if variable?
        get_all_properties_as_dict  - for saving to log

    Properties available to psi:
        runclass
        Target_count, background_count  - range of idx
        Channel count – how many speakers (2?)
        Current_target_location (for behavior assessment)
        Trial duration
        parameters available to user—depends on stimuli
        information for aligning to backgrounds? Is this useful/necessary?

    Trial structure

    1. ITI.
        Maintain background queue if – if there is one
        Pick new bgs as needed
        Get foreground info, target location

    2. Nose poke -> start trial
        Play foreground sound from speaker 1 and/or 2, play for Y seconds (Y<X?)
        Response = lick spout 1 or 2
        Timeout after Y seconds

    3. Loop to 1


    Natural streams – 2AFC
        Backgrounds have natural bg statistics
        Targets have natural fg statistics.

    Tone in noise – 2AFC
        No bg
        Target – one or two noises with varying relationship, tone embedded in one or the other.
        Or go/no-go?

    Phonemes
        No bg
        Target one or two phonemes, go/no-go
    """

    def __init__(self, FgSet=None, BgSet=None, combinations='simple',
                 fg_switch_channels=False, fg_delay=1.0,
                 random_seed=0):
        """
        :param FgSet: {WavFileSet, None}
        :param BgSet: {WavFileSet, None}
        :param combinations: str
            simple:
            all: every possible combination of bg and fg
        :param fg_switch_channels: bool
        :param fg_delay: float
        :param random_seed: int
        """
        if FgSet is None:
            self.FgSet = WaveformSet()
        else:
            self.FgSet = FgSet
        if BgSet is None:
            self.BgSet = WaveformSet()
        else:
            self.BgSet = BgSet
        self.combinations = combinations
        self.fg_switch_channels = fg_switch_channels
        self.fg_delay = fg_delay
        self.random_seed = random_seed
        self.current_trial = -1
        self.update()
        
    def update(self):
        """figure out indexing to map trial idx to specific members of FgSet and BgSet"""
        _rng = np.random.RandomState(self.random_seed)

        bg_range = np.arange(self.BgSet.max_index, dtype=int)
        fg_range = np.arange(self.FgSet.max_index, dtype=int)
        if self.fg_switch_channels:
            fg_channel = np.concatenate((np.zeros_like(fg_range), np.ones_like(fg_range)))
            fg_range = np.concatenate((fg_range, fg_range))
        else:
            fg_channel = np.zeros_like(fg_range)

        bg_len = len(bg_range)
        fg_len = len(fg_range)
        bgi = np.array([], dtype=int)
        fgi = np.array([], dtype=int)
        fgc = np.array([], dtype=int)
        if self.combinations == 'simple':
            total_trials = np.max([bg_len, fg_len, self.current_trial])
            print(f'Updating FgBgSet {total_trials} trials...')
            while (bg_len>0) & (len(bgi)<total_trials):
                bgi = np.concatenate((bgi, _rng.permutation(bg_range)))
            while (fg_len>0) & (len(fgi)<total_trials):
                ii = _rng.permutation(np.arange(len(fg_range)))
                fgi = np.concatenate((fgi, fg_range[ii]))
                fgc = np.concatenate((fgc, fg_channel[ii]))
        else:
            raise ValueError(f"FgBgSet combinations format {self.combinations} not supported")

        self.bg_index = bgi
        self.fg_index = fgi
        self.fg_channel = fgc


    def get_background(self, trialidx=None):
        if trialidx is None:
            trialidx = self.current_trial


    def get_foreground(self, trialidx=None):
        if trialidx is None:
            trialidx = self.current_trial


    def get_trial_waveform(self, trialidx=None):
        if trialidx is None:
            self.current_trial += 1
            trialidx = self.current_trial

        wfg = self.FgSet.waveform(self.fg_index[trialidx])
        cfg = self.fg_channel[trialidx]
        if cfg>0:
            wfg = np.concatenate((np.zeros_like(wfg), wfg), axis=1)
        wbg = self.BgSet.waveform(self.bg_index[trialidx])
        if wbg.shape[1]<wfg.shape[1]:
            wbg = np.concatenate((wbg, np.zeros_like(wbg)), axis=1)
        if wfg.shape[1]<wbg.shape[1]:
            wfg = np.concatenate((wfg, np.zeros_like(wfg)), axis=1)
        offsetbins = int(self.fg_delay * self.FgSet.fs)
        w = wbg
        w[offsetbins:(offsetbins+wfg.shape[0]),:] += wfg
        return w