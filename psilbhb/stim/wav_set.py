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
    elif level is not None:
        attenuatedB = 80-level
        sf = 10 ** (-attenuatedB/20)
        waveform *= (5 * sf)  # 5V RMS = 80 dB
        #log.info(f"Atten: {attenuatedB} SF: {sf} RMS: {waveform.std()}")

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

    def __init__(self, filenames, normalization='pe', level=65,
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
        self.level = level
        super().__init__(level=level, **kwargs)

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

    def waveform_zero(self, channel_count=None):
        if channel_count is None:
            files = self.filenames[0]
            if type(files) is str:
                files = [files]
            channel_count = len(files)
        if self.force_duration is None:
            force_duration = 0
        else:
            force_duration = self.force_duration

        sample_count = int(self.fs * force_duration)

        return np.zeros((sample_count, channel_count))

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
                 level=65, **kwargs):
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

        #log.info("**************************************************")
        #log.info(path)
        all_wav = list(sorted(Path(path).glob('*.wav')))
        #log.info(f"{list(all_wav)}")
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
        self.level = level

        #log.info(f"{filenames}")

        super().__init__(filenames, level=level, channel_count=channel_count, force_duration=force_duration, **kwargs)

class WavSet:
    pass

class FgBgSet(WavSet):
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
                 catch_frequency=0, primary_channel=0,
                 fg_delay=1.0, fg_snr=0.0, response_window=None,
                 migrate_fraction=0.0, migrate_start=0.5, migrate_stop=1.0,
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
        :param catch_frequency: float
            TODO: add fraction catch_frequency of trials with no target on top of
            regular trial set
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
        self.catch_frequency = catch_frequency
        self._fg_delay = fg_delay
        self._fg_snr = fg_snr
        self.primary_channel = primary_channel
        self.migrate_fraction = migrate_fraction
        self.migrate_start = migrate_start
        self.migrate_stop = migrate_stop

        if response_window is None:
            self.response_window = (0.0, 1.0)
        else:
            self.response_window = response_window
        self.random_seed = random_seed
        self.current_trial_idx = -1

        # trial management
        self.trial_wav_idx = np.array([], dtype=int)
        self.trial_outcomes = np.array([], dtype=int)
        self.trial_is_repeat = np.array([], dtype=int)
        self.current_full_rep = 0

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
            fg_channel = np.zeros_like(fg_range) + self.primary_channel
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
            #bg_range = np.tile(bg_range, 2)
        elif self.bg_switch_channels == 'combinatorial+diotic':
            bg_channel = np.concatenate((np.zeros_like(fg_channel), np.ones_like(fg_channel), -np.ones_like(fg_channel)))
            fg_channel = np.tile(fg_channel, 3)
            fg_range = np.tile(fg_range, 3)
            #bg_range = np.tile(bg_range, 3)
        else:
            raise ValueError(f"Unknown bg_switch_channels value: {self.bg_switch_channels}.")

        if (type(self._fg_snr) is np.array) | (type(self._fg_snr) is list):
            # multiple SNRs requested, tile
            fg_snr = np.concatenate(
                [np.zeros(len(fg_range)) + snr for snr in self._fg_snr]
            )
            bg_channel = np.tile(bg_channel, len(self._fg_snr))
            fg_channel = np.tile(fg_channel, len(self._fg_snr))
            fg_range = np.tile(fg_range, len(self._fg_snr))
            #bg_range = np.tile(bg_range, len(self._fg_snr))
        else:
            fg_snr = np.zeros(len(fg_range)) + self._fg_snr

        if self.catch_frequency>0:
            raise ValueError(f"Support for catch_frequency>0 not yet implemented")

        bg_len = len(bg_range)
        fg_len = len(fg_range)
        print(fg_len, bg_len)
        bgi = np.array([], dtype=int)
        fgi = np.array([], dtype=int)
        fgc = np.array([], dtype=int)
        bgc = np.array([], dtype=int)
        fgg = np.array([], dtype=int)
        fsnr = np.array([], dtype=int)
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
                fgg = np.concatenate((fgg, np.ones(len(fg_range))))
        elif self.combinations == 'all':
            total_wav_set = bg_len*fg_len
            for i,bg in enumerate(bg_range):
                bgi = np.concatenate((bgi, np.ones(len(fg_range), dtype=int)*bg))
                ii = np.arange(len(fg_range), dtype=int)
                fgi = np.concatenate((fgi, fg_range))
                fgc = np.concatenate((fgc, fg_channel[ii]))
                bgc = np.concatenate((bgc, bg_channel[ii]))
                fsnr = np.concatenate((fsnr, fg_snr[ii]))
                fgg = np.concatenate((fgg, np.ones(len(fg_range))))

        else:
            raise ValueError(f"FgBgSet combinations format {self.combinations} not supported")

        # remove redundant very high and very low SNR trials
        fgimin = fgi.min()
        bgimin = bgi.min()
        bgcmax = bgc.max()
        snr_keep = ((fsnr<=50) | ((bgi==bgimin) & (bgc==bgcmax))) & \
                   ((fsnr>-100) | (fgi==fgimin))
        bgi = bgi[snr_keep]
        fgi = fgi[snr_keep]
        bgc = bgc[snr_keep]
        fgc = fgc[snr_keep]
        fsnr = fsnr[snr_keep]
        fgg = fgg[snr_keep]
        fgg[fsnr<=-100]=-1

        migrate_keep = (fsnr>-30) & (bgc==bgcmax)

        if self.migrate_fraction>=1:
            migrate_trial = np.ones_like(fgi)
        elif (self.migrate_fraction > 0.4):
            migrate_trial = np.concatenate((np.zeros_like(fgi), np.ones_like(fgi[migrate_keep])))
            bgi = np.concatenate((bgi, bgi[migrate_keep]))
            fgi = np.concatenate((fgi, fgi[migrate_keep]))
            bgc = np.concatenate((bgc, bgc[migrate_keep]))
            fgc = np.concatenate((fgc, fgc[migrate_keep]))
            fsnr = np.concatenate((fsnr, fsnr[migrate_keep]))
            fgg = np.concatenate((fgg, fgg[migrate_keep]))
        elif (self.migrate_fraction > 0):
            migrate_trial = np.concatenate((np.zeros_like(fgi), np.zeros_like(fgi), np.ones_like(fgi[migrate_keep])))
            bgi = np.concatenate((bgi, bgi, bgi[migrate_keep]))
            fgi = np.concatenate((fgi, fgi, fgi[migrate_keep]))
            bgc = np.concatenate((bgc, bgc, bgc[migrate_keep]))
            fgc = np.concatenate((fgc, fgc, fgc[migrate_keep]))
            fsnr = np.concatenate((fsnr, fsnr, fsnr[migrate_keep]))
            fgg = np.concatenate((fgg, fgg, fgg[migrate_keep]))
        else:
            migrate_trial = np.zeros_like(fgi)

        total_wav_set = len(fgg)

        self.bg_index = bgi
        self.fg_index = fgi
        self.fg_channel = fgc
        self.bg_channel = bgc
        self.fg_snr = fsnr
        self.fg_go = fgg
        self.migrate_trial = migrate_trial

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
            self.current_full_rep += 1
            self.trial_is_repeat = np.concatenate((self.trial_is_repeat, np.zeros_like(new_trial_wav)))

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
        log.info(f"fg level: {self.FgSet.level} bg level: {self.BgSet.level} FG RMS: {wfg.std():.3f} BG RMS: {wbg.std():.3f}")

        if self.bg_channel[wav_set_idx] == 1:
            wbg = np.concatenate((np.zeros_like(wbg), wbg), axis=1)
        elif self.bg_channel[wav_set_idx] == -1:
            wbg = np.concatenate((wbg, wbg), axis=1)

        if wbg.shape[1] < wfg.shape[1]:
            wbg = np.concatenate((wbg, np.zeros_like(wbg)), axis=1)
        if wfg.shape[1] < wbg.shape[1]:
            wfg = np.concatenate((wfg, np.zeros_like(wfg)), axis=1)
        fg_snr = self.fg_snr[wav_set_idx]
        if fg_snr == -100:
            fg_scale = 0
        elif fg_snr < 50:
            fg_scale = 10**(fg_snr / 20)
        else:
            # special case of effectively infinite SNR, don't actually amplify fg
            wbg[:] = 0
            fg_scale = 10**((fg_snr-100) / 20)
        offsetbins = int(self.fg_delay[self.fg_index[wav_set_idx]] * self.FgSet.fs)

        if self.migrate_trial[wav_set_idx]:
            log.info('this is a target migration trial')
            start_bin = int(self.migrate_start*self.FgSet.fs)
            stop_bin = int(self.migrate_stop*self.FgSet.fs)
            end_mask = np.concatenate((np.zeros(start_bin),np.linspace(0,1,stop_bin-start_bin),
                                       np.ones(wfg.shape[0]-stop_bin)))[:,np.newaxis]
            start_mask = 1-end_mask
            w1 = np.fliplr(wfg) * start_mask
            #w2 = wfg
            w2 = wfg * end_mask
            wfg = w1 + w2

        # combine fg and bg waveforms
        w = wbg
        if wfg.shape[0]+offsetbins > wbg.shape[0]:
            print(wfg.shape[0], offsetbins, wbg.shape[0])
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
        if is_go_trial==-1:
            # -1 means either port
            response_condition = -1
        elif is_go_trial==1:
            # 1=spout 1, 2=spout 2
            response_condition = int(self.fg_channel[wav_set_idx]+1)
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
             'snr': self.fg_snr[wav_set_idx],
             'this_snr': self.fg_snr[wav_set_idx],
             'fg_delay': self.fg_delay[fg_i],
             'fg_channel': self.fg_channel[wav_set_idx],
             'bg_channel': self.bg_channel[wav_set_idx],
             'migrate_trial': self.migrate_trial[wav_set_idx],
             'response_condition': response_condition,
             'response_window': response_window,
             'current_full_rep': self.current_full_rep,
             'primary_channel': self.primary_channel,
             'trial_is_repeat': self.trial_is_repeat[trial_idx],
             }
        return d

    def score_response(self, outcome, repeat_incorrect=2, trial_idx=None):
        """
        current logic: if invalid or incorrect, trial should be repeated
        :param outcome: int
            -1 trial not scored (yet?) - happens if score_response skips a trial_idx
            0 invalid
            1 incorrect
            2 correct
        :param repeat_incorrect: no/early/all
            If all -- repeat all incorrect, if early, repeat only early withdraws
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
        if (repeat_incorrect==2 and (outcome in [0, 1])) or (repeat_incorrect==1 and (outcome in [0])):
            #log.info('Trial {trial_idx} outcome {outcome}: appending repeat to trial_wav_idx')
            #self.trial_wav_idx = np.concatenate((self.trial_wav_idx, [self.trial_wav_idx[trial_idx]]))
            #self.trial_is_repeat = np.concatenate((self.trial_is_repeat, [1]))
            # log.info('Trial {trial_idx} outcome {outcome}: appending repeat to trial_wav_idx')
            # self.trial_wav_idx = np.concatenate((self.trial_wav_idx, [self.trial_wav_idx[trial_idx]]))
            # self.trial_is_repeat = np.concatenate((self.trial_is_repeat, [1]))
            log.info('Trial {trial_idx} outcome {outcome}: repeating immediately')
            self.trial_wav_idx = np.concatenate((self.trial_wav_idx[:trial_idx],
                                                 [self.trial_wav_idx[trial_idx]],
                                                 self.trial_wav_idx[trial_idx:]))
            self.trial_is_repeat = np.concatenate((self.trial_is_repeat[:(trial_idx + 1)],
                                                   [1],
                                                   self.trial_is_repeat[(trial_idx + 1):]))

        else:
            log.info('Trial {trial_idx} outcome {outcome}: moving on')

"""
        sound_path=params['sound_path'],
        target_set=params['target_set'],
        non_target_set=params['non_target_set'],
        catch_set=params['catch_set'],
        switch_channels=params['switch_channels'], 
        primary_channel=params['primary_channel'], 
        duration=params['duration'],
        repeat_count=params['repeat_count'],
        repeat_isi=params['repeat_isi'], 
        tar_to_cat_ratio=params['tar_to_cat_ratio'],
        level=params['level'], 
        fs=params['fs'], 
        response_start=params['response_start'], 
        response_end=params['response_end'], 
        random_seed=params['random_seed'])
"""
class VowelSet(WavSet):

    def __init__(self, sound_path='/auto/data/sounds/vowels/v2/',
                 target_set=['EE_106'],
                 non_target_set=['IH_106'],
                 catch_set=[],
                 switch_channels=False, primary_channel=0, repeat_count=1,
                 repeat_isi=0.2, tar_to_cat_ratio=5,
                 level=60, duration=0.24, fs=44000,
                 response_start=0, response_end=1, random_seed=0, n_response=2):

        # internal object to handle wavs, don't need to specify independently
        self.wavset = MCWavFileSet(
            fs=fs, path=sound_path, duration=duration, normalization='rms',
            fit_range=slice(0, None), test_range=None, test_reps=2,
            channel_count=1, level=level)
        self.target_set = target_set
        self.non_target_set = non_target_set
        self.catch_set = catch_set
        self.switch_channels = switch_channels
        self.primary_channel = primary_channel

        self.repeat_count = repeat_count
        self.repeat_isi = repeat_isi
        self.tar_to_cat_ratio = tar_to_cat_ratio
        self.random_seed = random_seed
        self.response_window = [response_start, response_end]

        self.n_response = n_response
        log.info('N_response %r', self.n_response)
        self.current_trial_idx = -1
        self.duration = 0

        # trial management
        self.trial_wav_idx = np.array([], dtype=int)
        self.trial_outcomes = np.array([], dtype=int)
        self.trial_is_repeat = np.array([], dtype=int)
        self.current_full_rep = 0

        self.update()

    @property
    def wav_per_rep(self):
        return len(self.stim1idx)

    def update(self, trial_idx=None):
        """figure out indexing to map wav_set idx to specific members of FgSet and BgSet.
        manage trials separately to allow for repeats, etc."""
        _rng = np.random.RandomState(self.random_seed)
        ttc=self.tar_to_cat_ratio
        all_stim = self.target_set*ttc + self.non_target_set*ttc + self.catch_set
        all_cat = ['T'] * len(self.target_set)*ttc + \
                  ['N'] * len(self.non_target_set)*ttc + \
            ['C'] * len(self.catch_set)
        stim1 = [(s+'+').split('+')[0] for s in all_stim]
        stim2 = [(s+'+').split('+')[1] for s in all_stim]
        names = self.wavset.names

        if self.primary_channel==0:
            stim1idx = [[i for i in names if (s!='') & i.startswith(s)][slice(0,1)] for s in stim1]
            stim2idx = [[i for i in names if (s!='') & i.startswith(s)][slice(0,1)] for s in stim2]
        else:
            stim1idx = [[i for i in names if (s!='') & i.startswith(s)][slice(0,1)] for s in stim2]
            stim2idx = [[i for i in names if (s!='') & i.startswith(s)][slice(0,1)] for s in stim1]

        stim1idx = [names.index(s[0]) if len(s)>0 else -1 for s in stim1idx]
        stim2idx = [names.index(s[0]) if len(s)>0 else -1 for s in stim2idx]
        if self.switch_channels:
            self.stim1idx = stim1idx + stim2idx
            self.stim2idx = stim2idx + stim1idx
            self.stim_cat = all_cat * 2
        else:
            self.stim1idx = stim1idx
            self.stim2idx = stim2idx
            self.stim_cat = all_cat
        self.duration = self.repeat_count * self.wavset.duration + \
                        (self.repeat_count-1) * self.repeat_isi
        # set up wav_set_idx to trial_idx mapping  -- self.trial_wav_idx
        if trial_idx is None:
            trial_idx = self.current_trial_idx
        if trial_idx >= len(self.trial_wav_idx):
            for rep in np.arange(self.current_full_rep+1):
                new_trial_wav = _rng.permutation(np.arange(len(self.stim1idx), dtype=int))
            self.trial_wav_idx = np.concatenate((self.trial_wav_idx, new_trial_wav))
            log.info(f'Added {len(new_trial_wav)}/{len(self.trial_wav_idx)} trials to trial_wav_idx')
            self.current_full_rep += 1
            self.trial_is_repeat = np.concatenate((self.trial_is_repeat, np.zeros_like(new_trial_wav)))

    def trial_waveform(self, trial_idx=None, wav_set_idx=None):
        if wav_set_idx is None:
            if trial_idx is None:
                self.current_trial_idx += 1
                trial_idx = self.current_trial_idx
            if len(self.trial_wav_idx) <= trial_idx:
                self.update(trial_idx=trial_idx)

            wav_set_idx = self.trial_wav_idx[trial_idx]

        s1idx = self.stim1idx[wav_set_idx]
        if s1idx >= 0:
            w1 = self.wavset.waveform(s1idx)
        else:
            w1 = self.wavset.waveform_zero()
        s2idx = self.stim2idx[wav_set_idx]
        if s2idx >= 0:
            w2 = self.wavset.waveform(s2idx)
        else:
            w2 = self.wavset.waveform_zero()
        w = np.concatenate([w1, w2], axis=1)
        
        if self.repeat_count>1:
            isi_bins = int(self.wavset.fs * self.repeat_isi)
            w_silence=np.zeros((isi_bins, w.shape[1]))
            w_all = [w] + [w_silence, w] * (self.repeat_count-1)
            w = np.concatenate(w_all, axis=0)

        return w.T

    def trial_parameters(self, trial_idx=None, wav_set_idx=None):
        """
        :param trial_idx:
        :param wav_set_idx:
        :return: dict
           'response_condition': (1: spout 1, 2: spout 2, -1: either spout)
        """
        if wav_set_idx is None:
            if trial_idx is None:
                self.current_trial_idx += 1
                trial_idx = self.current_trial_idx
            if len(self.trial_wav_idx) <= trial_idx:
                self.update(trial_idx=trial_idx)

            wav_set_idx = self.trial_wav_idx[trial_idx]
        else:
            trial_idx = 0

        s1idx = self.stim1idx[wav_set_idx]
        s2idx = self.stim2idx[wav_set_idx]
        if s1idx>0:
            s1_name = self.wavset.names[s1idx]
        else:
            s1_name = ''
        if s2idx>0:
            s2_name = self.wavset.names[s2idx]
        else:
            s2_name = ''

        stim_cat = self.stim_cat[wav_set_idx]

        if self.n_response == 2:
            if stim_cat == 'T':
                response_condition = 1
            elif stim_cat == 'N':
                response_condition = 2
            elif stim_cat == 'C':
                response_condition = -1
        elif self.n_response == 1:
            if stim_cat == 'T':
                response_condition = 1
            elif stim_cat == 'N':
                response_condition = 0
            elif stim_cat == 'C':
                response_condition = 0

        response_window = self.response_window

        d = {'trial_idx': trial_idx,
             'wav_set_idx': wav_set_idx,
             's1idx': s1idx,
             's2idx': s2idx,
             's1_name': s1_name,
             's2_name': s2_name,
             'duration': self.duration,
             'response_condition': response_condition,
             'response_window': response_window,
             'current_full_rep': self.current_full_rep,
             'primary_channel': self.primary_channel,
             'trial_is_repeat': self.trial_is_repeat[trial_idx],
             }
        return d

    def score_response(self, outcome, repeat_incorrect=True, trial_idx=None):
        """
        current logic: if invalid or incorrect, trial should be repeated
        :param outcome: int
            -1 trial not scored (yet?) - happens if score_response skips a trial_idx
            0 invalid
            1 incorrect
            2 correct
            3 correct - either response ok
        :param repeat_incorrect: bool
            If True, repeat incorrect and invalid trials.
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
        if repeat_incorrect and (outcome in [0, 1]):
            #log.info('Trial {trial_idx} outcome {outcome}: appending repeat to trial_wav_idx')
            #self.trial_wav_idx = np.concatenate((self.trial_wav_idx, [self.trial_wav_idx[trial_idx]]))
            #self.trial_is_repeat = np.concatenate((self.trial_is_repeat, [1]))
            log.info('Trial {trial_idx} outcome {outcome}: repeating immediately')
            self.trial_wav_idx = np.concatenate((self.trial_wav_idx[:trial_idx],
                                                 [self.trial_wav_idx[trial_idx]],
                                                 self.trial_wav_idx[trial_idx:]))
            self.trial_is_repeat = np.concatenate((self.trial_is_repeat[:(trial_idx+1)],
                                                 [1],
                                                 self.trial_is_repeat[(trial_idx+1):]))
        else:
            log.info('Trial {trial_idx} outcome {outcome}: moving on')
