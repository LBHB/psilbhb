
from functools import partial
import itertools
from pathlib import Path

import numpy as np
from scipy import signal
from scipy.io import wavfile

from psiaudio import queue
from psiaudio import util

from psiaudio.stim import Waveform, FixedWaveform, ToneFactory, \
    WavFileFactory, WavSequenceFactory, wavs_from_path

import logging
log = logging.getLogger(__name__)


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


class WaveformSet():
    def __init__(self, fs=40000, level=65, calibration=None, channel_count=1):
        """

        :param fs:  float
            Sampling rate of output. If wav file sampling rate is different, it
            will be resampled to the correct sampling rate using a FFT-based
            resampling algorithm.
        :param level: float
            Level to present wav files at (currently peSPL due to how the
            normalization works).
        :param calibration: instance of Calibration
            Used to scale waveform to appropriate peSPL. If not provided,
            waveform is not scaled.
        :param channel_count: int

        """
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
        :param normalization: {'pe', 'rms', 'fixed'}
            Method for rescaling waveform. If `'pe'`, rescales to peak-equivalent
            so the max value of the waveform matches the target level. If `'rms'`,
            rescales so that the RMS value of the waveform matches the target
            level. If 'fixed', scale by a fixed value (norm_fixed_scale)
        :param norm_fixed_scale: float
            if normalization=='fixed', multiply the wavform by this value.
        :param force_duration: {None, float}
            Duration of each wav file. If None, the wav file is loaded at the
            beginning so that durations can be established. For large
            directories, this can slow down the startup time of the program.
            Knowing the exact duration may be important for some downstream
            operations. For example, epoch extraction relative to the
            presentation time of a particular wav file; estimating the overall
            duration of the entire wav sequence, etc.  If you don't have a need
            for these operations and want to speed up loading of wav files, set
            this value to -1 (the default).
        :param random_seed: int
            Random seed to use when initializing queue for random sequence.
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


class MCWavFileSet(WavFileSet):

    def __init__(self, path, duration=-1, include_silence=True,
                 fit_range=None, fit_reps=1, test_range=None, test_reps=0,
                 channel_count=1, channel_offset=1, binaural_combinations='single_offset',
                 **kwargs):
        """
        :param path:
            Path to directory containing wav files.
        :param duration: float
            duration in sec. Truncate or zero pad as needed
        :param include_silence:
            include a silent waveform in the pool
        :param fit_range: {None, slice, list}
        :param fit_reps: int
        :param test_range: {None, slice, list}
        :param test_reps: int
        :param channel_count: int
        :param channel_offset: int
        :param binaural_combinations: str
        :param kwargs: passthrough parameters to WavefileSet
       """
        '''
        Parameters
        ----------
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
        
        if type(fit_range) is slice:
            fit_range = list(range(len(all_wav))[fit_range])
        if type(test_range) is slice:
            test_range = list(range(len(all_wav))[test_range])
            
        if include_silence:
            silent_f = [f for i,f in enumerate(all_wav) if str(f).endswith('x_silence.wav')]
            ff = [all_wav[f] for f in fit_range] + silent_f
            tt = [all_wav[f] for f in test_range] + silent_f
        else:
            ff = [all_wav[f] for f in fit_range]
            tt = [all_wav[f] for f in test_range]

        fit_wav = []
        test_wav = []
        wav = []
        if channel_count == 1:
            fit_wav = [[_f] for _f in ff] * fit_reps
            test_wav = [[_t] for _t in tt] * test_reps
            filenames = fit_wav + test_wav

        elif binaural_combinations == 'single_offset':
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
        self.fit_range = fit_range
        self.test_range = test_range

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
                 fg_switch_channels=False, bg_switch_channels=False, 
                 fg_go_index=None,
                 fg_delay=1.0, fg_snr=0.0, response_window=None,
                 random_seed=0):
        """
        FgBgSet polls FgSet and BgSet for .max_index, .waveform, .names, and .fs
        :param FgSet: {WaveformSet, MultichannelWaveformSet, None}
        :param BgSet: {WaveformSet, MultichannelWaveformSet, None}
        :param combinations: str
            simple: random pairing
            all: exhaustive, every possible combination of bg and fg
        :param fg_switch_channels: bool
        :param bg_switch_channels: {bool, str}
            channel of bg relative to fg
            False: just play straight as BgSet provides them (for Bg managing binaural)
            same: same as fg
            combinatorial: same + opposite
            opposite: opposite to fg only
        :param fg_go_index: {list,np.array}
            TODO: allow list of "go" indices?
            0 or 1 for each entry in FgSet. len>=FgSet.max_index
        :param fg_delay: float
        :param response_window: {tuple, np.array, None}
            None converts to (0,1) = 0 to 1 sec after fg starts playing
            If array, length should bg >= FgSet.max_index
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
        self.bg_switch_channels = bg_switch_channels
        self.fg_go_index = fg_go_index
        self._fg_delay = fg_delay
        self._fg_snr = fg_snr
        if response_window is None:
            self.response_window = (0.0, 1.0)
        else:
            self.response_window = response_window
        self.random_seed = random_seed
        self.current_trial_idx = -1

        # trial management
        self.trial_wav_idx = np.array([], dtype=int)
        self.trial_outcomes = np.array([], dtype=int)
        self.current_repetition = 0

        self.update()

    @property
    def wav_per_rep(self):
        return np.min([len(self.bg_index), len(self.fg_index)])

    def update(self, trial_idx=None):
        """figure out indexing to map wav_set idx to specific members of FgSet and BgSet.
        manage trials separately to allow for repeats, etc."""
        _rng = np.random.RandomState(self.random_seed)

        bg_range = np.arange(self.BgSet.max_index, dtype=int)
        fg_range = np.arange(self.FgSet.max_index, dtype=int)
        if self.fg_switch_channels:
            fg_channel = np.concatenate((np.zeros_like(fg_range), np.ones_like(fg_range)))
            fg_range = np.tile(fg_range, 2)
        else:
            fg_channel = np.zeros_like(fg_range)
        if self.bg_switch_channels == False:
            bg_channel = np.zeros_like(fg_range)
        elif self.bg_switch_channels == 'same':
            bg_channel = fg_channel.copy()
        elif self.bg_switch_channels == 'opposite':
            bg_channel = 1-fg_channel
        elif self.bg_switch_channels == 'combinatorial':
            bg_channel = np.concatenate((np.zeros_like(fg_channel), np.ones_like(fg_channel)))
            fg_channel = np.tile(fg_channel, 2)
            fg_range = np.tile(fg_range, 2)
            bg_range = np.tile(bg_range, 2)
        else:
            raise ValueError(f"Unknown bg_switch_channels value: {self.bg_switch_channels}.")
        if self.fg_go_index is not None:
            fg_go = self.fg_go_index
        else:
            fg_go = np.ones_like(fg_range)

        bg_len = len(bg_range)
        fg_len = len(fg_range)
        bgi = np.array([], dtype=int)
        fgi = np.array([], dtype=int)
        fgc = np.array([], dtype=int)
        bgc = np.array([], dtype=int)
        fgg = np.array([], dtype=int)
        if self.combinations == 'simple':
            total_wav_set = np.max([bg_len, fg_len])
            #print(f'Updating FgBgSet {total_wav_set} trials...')
            while (bg_len>0) & (len(bgi) < total_wav_set):
                #bgi = np.concatenate((bgi, _rng.permutation(bg_range)))
                bgi = np.concatenate((bgi, bg_range))
            while (fg_len>0) & (len(fgi) < total_wav_set):
                #ii = _rng.permutation(np.arange(len(fg_range)))
                ii = np.arange(len(fg_range))
                fgi = np.concatenate((fgi, fg_range))
                fgc = np.concatenate((fgc, fg_channel[ii]))
                bgc = np.concatenate((bgc, bg_channel[ii]))
                fgg = np.concatenate((fgg, fg_go[ii]))
        elif self.combinations == 'all':
            total_wav_set = bg_len*fg_len
            for i,bg in enumerate(bg_range):
                bgi = np.concatenate((bgi, np.ones(len(fg_range), dtype=int)*bg))
                ii = np.arange(len(fg_range), dtype=int)
                fgi = np.concatenate((fgi, fg_range))
                fgc = np.concatenate((fgc, fg_channel[ii]))
                bgc = np.concatenate((bgc, bg_channel[ii]))
                fgg = np.concatenate((fgg, fg_go[ii]))

        else:
            raise ValueError(f"FgBgSet combinations format {self.combinations} not supported")

        self.bg_index = bgi
        self.fg_index = fgi
        self.fg_channel = fgc
        self.bg_channel = bgc
        self.fg_go = fgc

        if (type(self._fg_snr) is np.array) | (type(self._fg_snr) is list):
            self.fg_snr = np.array(self._fg_snr)
        else:
            self.fg_snr = np.zeros(self.FgSet.max_index) + self._fg_snr
        if (type(self._fg_delay) is np.array) | (type(self._fg_delay) is list):
            self.fg_delay = np.array(self._fg_delay)
        else:
            self.fg_delay = np.zeros(self.FgSet.max_index) + self._fg_delay

        # set up wav_set_idx to trial_idx mapping  -- self.trial_wav_idx
        if trial_idx is None:
            trial_idx = self.current_trial_idx
        if trial_idx >= len(self.trial_wav_idx):
            new_trial_wav = _rng.permutation(np.arange(total_wav_set, dtype=int))
            self.trial_wav_idx = np.concatenate((self.trial_wav_idx, new_trial_wav))
            log.info(f'Added {len(new_trial_wav)}/{len(self.trial_wav_idx)} trials to trial_wav_idx')
            self.current_repetition += 1

    def trial_waveform(self, trial_idx=None, wav_set_idx=None):
        if wav_set_idx is None:
            if trial_idx is None:
                self.current_trial_idx += 1
                trial_idx = self.current_trial_idx
            if len(self.trial_wav_idx) <= trial_idx:
                self.update(trial_idx=trial_idx)

            wav_set_idx = self.trial_wav_idx[trial_idx]

        wfg = self.FgSet.waveform(self.fg_index[wav_set_idx])
        if self.fg_channel[wav_set_idx] == 1:
            wfg = np.concatenate((np.zeros_like(wfg), wfg), axis=1)
        wbg = self.BgSet.waveform(self.bg_index[wav_set_idx])
        if self.bg_channel[wav_set_idx] == 1:
            wbg = np.concatenate((np.zeros_like(wbg), wbg), axis=1)
        if wbg.shape[1] < wfg.shape[1]:
            wbg = np.concatenate((wbg, np.zeros_like(wbg)), axis=1)
        if wfg.shape[1] < wbg.shape[1]:
            wfg = np.concatenate((wfg, np.zeros_like(wfg)), axis=1)
        fg_snr = self.fg_snr[self.fg_index[wav_set_idx]]
        fg_scale = 10**(fg_snr / 20)
        offsetbins = int(self.fg_delay[self.fg_index[wav_set_idx]] * self.FgSet.fs)

        # combine fg and bg waveforms
        w = wbg
        if wfg.shape[0]+offsetbins > wbg.shape[0]:
            print(wfg.shape[0], offsetbins , wbg.shape[0])
            w = np.concatenate((w, np.zeros((wfg.shape[0]+offsetbins-wbg.shape[0],
                                             wbg.shape[1]))), axis=0)
        w[offsetbins:(offsetbins+wfg.shape[0]), :] += wfg * fg_scale
        if w.shape[1] < 2:
            w = np.concatenate((w, np.zeros_like(w)), axis=1)
        return w.T

    def trial_parameters(self, trial_idx=None, wav_set_idx=None):
        if wav_set_idx is None:
            if trial_idx is None:
                trial_idx = self.current_trial_idx
            if len(self.trial_wav_idx) <= trial_idx:
                self.update(trial_idx=trial_idx)

            wav_set_idx = self.trial_wav_idx[trial_idx]
        else:
            trial_idx = 0

        fg_i = self.fg_index[wav_set_idx]
        bg_i = self.bg_index[wav_set_idx]

        is_go_trial = self.fg_go[wav_set_idx]
        if is_go_trial:
            # 1=spout 1, 2=spout 2
            response_condition = self.fg_channel[wav_set_idx]+1
        else:
            response_condition = 0

        if type(self.response_window) is tuple:
            response_window = (self.fg_delay[fg_i] + self.response_window[0],
                               self.fg_delay[fg_i] + self.response_window[1])
        else:
            response_window = (self.fg_delay[fg_i] + self.response_window[fg_i][0],
                               self.fg_delay[fg_i] + self.response_window[fg_i][1])

        d = {'trial_idx': trial_idx,
             'wav_set_idx': wav_set_idx,
             'fg_i': fg_i,
             'bg_i': bg_i,
             'fg_name': self.FgSet.names[fg_i],
             'bg_name': self.BgSet.names[bg_i],
             'fg_duration': self.FgSet.duration,
             'bg_duration': self.BgSet.duration,
             'fg_snr': self.fg_snr[fg_i],
             'fg_delay': self.fg_delay[fg_i],
             'fg_channel': self.fg_channel[wav_set_idx],
             'bg_channel': self.bg_channel[wav_set_idx],
             'response_condition': response_condition,
             'response_window': response_window,
             }
        return d

    def score_response(self, outcome, trial_idx=None):
        """
        current logic: if invalid or incorrect, trial should be repeated
        :param outcome: int
            -1 trial not scored (yet?) - happens if score_response skips a trial_idx
            0 invalid
            1 incorrect
            2 correct
        :param trial_idx: int
            must be less than len(trial_wav_idx) to be valid. by default, updates score for 
            current_trial_idx and increments current_trial_idx by 1.
        :return:
        """
        if trial_idx is None:
            trial_idx = self.current_trial_idx
            # Only incrementing current trial index if trial_idx is None. Do we always 
            # want to do this???
            self.current_trial_idx = trial_idx + 1

        if trial_idx>=len(self.trial_wav_idx):
            raise ValueError(f"attempting to score response for trial_idx out of range")

        if trial_idx>=len(self.trial_outcomes):
            n = trial_idx - len(self.trial_outcomes) + 1
            self.trial_outcomes = np.concatenate((self.trial_outcomes, np.zeros(n)-1))
        self.trial_outcomes[trial_idx] = int(outcome)
        if outcome in [0, 1]:
            log.info('Trial {trial_idx} outcome {outcome}: appending repeat to trial_wav_idx')
            self.trial_wav_idx = np.concatenate((self.trial_wav_idx, [self.trial_wav_idx[trial_idx]]))
        else:
            log.info('Trial {trial_idx} outcome {outcome}: moving on')

