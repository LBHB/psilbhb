from functools import partial
import itertools
from pathlib import Path

import numpy as np
from scipy import signal
from scipy.io import wavfile
import pandas as pd

from psiaudio import queue
from psiaudio import util
from fractions import Fraction
from copy import deepcopy
from random import choices
import os

import logging
log = logging.getLogger(__name__)

def get_stim_list(FgSet, BgSet, catch_ferret_id=3, n_env_bands=[2, 8, 32], reg2catch_ratio=7):
    be_verbose = 1
    if os.path.exists('h:/sounds'):
        soundpath_fg = 'h:/sounds/Categories/v3_vocoding'
        soundpath_bg = 'h:/sounds/Categories/speech_stims'
        soundpath_catch_bg = 'h:/sounds/Categories/chimeric_voc'
    else:
        soundpath_fg = '/auto/users/satya/code/projects_getting_started/explore_bignat/ferret_vocals/v3_vocoding'
        soundpath_bg = '/auto/users/satya/code/projects_getting_started/explore_bignat/ferret_vocals/speech_stims'
        soundpath_catch_bg = '/auto/users/satya/code/projects_getting_started/explore_bignat/ferret_vocals/chimeric_voc'

    all_ferret_files = [file for file in os.listdir(soundpath_fg) if file.endswith(".wav") and file.startswith("fer")]
    all_speech_files = [file for file in os.listdir(soundpath_bg) if file.endswith(".wav") and file.startswith("spe")]
    all_catch_files = [file for file in os.listdir(soundpath_catch_bg) if file.endswith(".wav") and file.startswith("ENV")]

    taboo_ferret_files = ['ferretb3001R.wav', 'ferretb4004R.wav']

    all_ferret_files = [x for x in all_ferret_files if x not in taboo_ferret_files]
    all_catch_files = [x for x in all_catch_files if x.split('_')[-1] not in taboo_ferret_files]


    #region get all catch trial files: fgs and env bgs
    catch_bgs = [x for x in all_catch_files if
                 any([x.startswith("ENV{}_ferretb{}".format(nb,catch_ferret_id)) for nb in n_env_bands]) ]
    catch_fgs = list(set([x.split('_')[-1] for x in catch_bgs]))
    catch_fgs.sort()

    catch_pair_fg_inds = np.arange(4)
    catch_pair_bg_inds = np.array([1, 0, 3, 2])

    catch_fg_inds = np.concatenate([np.tile(x, len(n_env_bands)) for x in catch_pair_fg_inds])
    catch_bg_inds = np.array([catch_pair_bg_inds[x] for x in catch_fg_inds])
    catch_env_nbands = np.tile(n_env_bands, len(catch_pair_fg_inds))
    num_catch_trials = len(catch_fg_inds)

    catch_fg_names = [soundpath_fg + '/' + catch_fgs[idx] for idx in catch_fg_inds]
    catch_bg_names = [soundpath_catch_bg + '/' + 'ENV{}_{}'.format(nb,catch_fgs[bg_idx]) for bg_idx,nb in zip(catch_bg_inds,catch_env_nbands)]

    if be_verbose:
        print("~~~~~~~~~Catch~~~~~~~~~")
        [print(catch_fg_names[i] + ' vs ' + catch_bg_names[i]) for i in range(num_catch_trials)]
        print("~~~~~~~~~end Catch~~~~~~~~~")
    #endregion

    #region get all regular (non-catch) trial stimuli: fgs and speech
    num_regular_trials= reg2catch_ratio*num_catch_trials

    # taboo_ferret_ids = [1, 2, 7, catch_ferret_id]
    taboo_ferret_ids = [1, 2, 7]
    # reg_fg_names = choices([soundpath_fg + '/' + x for x in all_ferret_files
    #                         if not any([x.startswith("ferretb{}".format(tabid)) for tabid in taboo_ferret_ids])],
    #                        k=num_regular_trials)
    reg_fg_names = np.random.choice([soundpath_fg + '/' + x for x in all_ferret_files
                                     if not any([x.startswith("ferretb{}".format(tabid)) for tabid in taboo_ferret_ids])],
                                    size=num_regular_trials)

    # reg_bg_names = [soundpath_bg + '/' + x for x in choices(all_speech_files, k=num_regular_trials)]
    reg_bg_names = [soundpath_bg + '/' + x for x in np.random.choice(all_speech_files, size=num_regular_trials)]
    if be_verbose:
        print("~~~~~~~~~Regular~~~~~~~~~")
        [print(reg_fg_names[i] + ' vs ' + reg_bg_names[i]) for i in range(num_regular_trials)]
        print("~~~~~~~~~end Regular~~~~~~~~~")
    #endregion

    session_fg_files = reg_fg_names + catch_fg_names
    session_bg_files = reg_bg_names + catch_bg_names

    # get indices
    wav_set_fg = [str(x[0]).replace('\\', '/') for x in FgSet.filenames]
    wav_set_bg = [str(x[0]).replace('\\', '/') for x in BgSet.filenames]

    fgi = np.array([wav_set_fg.index(x) for x in session_fg_files])
    bgi = np.array([wav_set_bg.index(x) for x in session_bg_files])
    fgg = np.concatenate((np.ones(num_regular_trials), -1*np.ones(num_catch_trials)))

    if be_verbose:
        session_num_trials = num_regular_trials + num_catch_trials
        print("~~~~~~~~~ Session ~~~~~~~~~")
        [print(f"{i+1}/{len(session_fg_files)}: fgg={fgg[i]} {session_fg_files[i]} vs {session_bg_files[i]}")
         for i in range(session_num_trials)]
        print("~~~~~~~~~end Session~~~~~~~~~")

    return fgi, bgi, fgg

def get_stim_list_no_catch(FgSet, BgSet):
    be_verbose = 0
    if os.path.exists('h:/sounds'):
        soundpath_fg = 'h:/sounds/Categories/v3_vocoding'
        soundpath_bg = 'h:/sounds/Categories/speech_stims'
        soundpath_overall_noise = 'h:/sounds/Categories/overall_noise'
    else:
        soundpath_fg = '/auto/users/satya/code/projects_getting_started/explore_bignat/ferret_vocals/v3_vocoding'
        soundpath_bg = '/auto/users/satya/code/projects_getting_started/explore_bignat/ferret_vocals/speech_stims'
        soundpath_overall_noise = '/auto/users/satya/code/projects_getting_started/explore_bignat/ferret_vocals/overall_noise'

    all_ferret_files = [file for file in os.listdir(soundpath_fg) if file.endswith(".wav") and file.startswith("fer")]
    all_speech_files = [file for file in os.listdir(soundpath_bg) if file.endswith(".wav") and file.startswith("spe")]
    all_noise_files = [file for file in os.listdir(soundpath_bg) if file.endswith(".wav") and file.startswith("spe")]


    taboo_ferret_files = ['ferretb3001R.wav', 'ferretb4004R.wav']
    all_ferret_files = [x for x in all_ferret_files if x not in taboo_ferret_files]

    #region get all regular (non-catch) trial stimuli: fgs and speech
    num_regular_trials= 100

    # taboo_ferret_ids = [1, 2, 7, catch_ferret_id]
    taboo_ferret_ids = [1, 2, 7]
    # reg_fg_names = choices([soundpath_fg + '/' + x for x in all_ferret_files
    #                         if not any([x.startswith("ferretb{}".format(tabid)) for tabid in taboo_ferret_ids])],
    #                        k=num_regular_trials)
    reg_fg_names = np.random.choice([soundpath_fg + '/' + x for x in all_ferret_files
                                     if not any([x.startswith("ferretb{}".format(tabid)) for tabid in taboo_ferret_ids])],
                                    size=num_regular_trials)
    # reg_bg_names = [soundpath_bg + '/' + x for x in choices(all_speech_files, k=num_regular_trials)]
    reg_bg_names = [soundpath_bg + '/' + x for x in np.random.choice(all_speech_files, size=num_regular_trials)]
    if be_verbose:
        print("~~~~~~~~~Regular~~~~~~~~~")
        [print(reg_fg_names[i] + ' vs ' + reg_bg_names[i]) for i in range(num_regular_trials)]
        print("~~~~~~~~~end Regular~~~~~~~~~")
    #endregion

    session_fg_files = reg_fg_names
    session_bg_files = reg_bg_names

    # get indices
    wav_set_fg = [str(x[0]).replace('\\', '/') for x in FgSet.filenames]
    wav_set_bg = [str(x[0]).replace('\\', '/') for x in BgSet.filenames]

    fgi = np.array([wav_set_fg.index(x) for x in session_fg_files])
    bgi = np.array([wav_set_bg.index(x) for x in session_bg_files])
    fgg = np.ones(num_regular_trials)

    if be_verbose:
        session_num_trials = num_regular_trials
        print("~~~~~~~~~ Session ~~~~~~~~~")
        [print(f"{i+1}/{len(session_fg_files)}: fgg={fgg[i]} {session_fg_files[i]} vs {session_bg_files[i]}")
         for i in range(session_num_trials)]
        print("~~~~~~~~~end Session~~~~~~~~~")

    return fgi, bgi, fgg


def cat_MCWavFileSets(set1, set2):
    """
        * Stand-alone function to concatenate two MCWavFileSets.
            - Probably make it a method for MCWavFileSets(instead of a function like this?)
            - Can create a more general function that takes in multiple (>=2) sets
        * Creating this function to combine BGCatchTrials with backgrounds (or FGCatchTrials with foregrounds)
            for 2-AFC FG-vs-BG discrimination.
        Inputs:
            :param set1: first merge_MCWavFileSets
            :param set2: second merge_MCWavFileSets

        Output:
            :param Concatenated MCWavFileSet

        Comments/Notes:
            Most fields (attributes) are being used either directly (e.g., fs, level)
                or indirectly inside methods (e.g., names, waveform). So go through all fields and apply some criteria
                to either check of compatibility or concatenate values.
    """

    # Attributes to check for compatibility (i.e., identical values)
    compatibility_fields = ['channel_offset', 'duration', 'level', 'normalization', 'norm_fixed_scale', 'force_duration',
                            'random_seed', 'fs', 'calibration', 'channel_count']
    # for dict_var in compatibility_fields:
    #     print(f"{dict_var}: {getattr(set1, dict_var)} vs {getattr(set2, dict_var)}")
    out_assert = np.array([getattr(set1, _atb)==getattr(set2, _atb) for _atb in compatibility_fields])
    assert all(out_assert), ("The following attributes do not match between the two sets ({})"
                             .format([compatibility_fields[idx] for idx,tf_bool in enumerate(out_assert) if not tf_bool]))

    cat_set = deepcopy(set1)
    concat_fields = ['fit_names', 'test_names', 'fit_range', 'test_range']
    for _cat_fld in concat_fields:
        setattr(cat_set, _cat_fld, getattr(cat_set, _cat_fld) + getattr(set2, _cat_fld))

    cat_set.filenames = set1.filenames + set2.filenames
    cat_set.filelabels = set1.filelabels + set2.filelabels

    # log.info(f"Entered cat fun: {nrep_set1}|{nrep_set2}")
    return cat_set
# End of cat_MCWavFileSets


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
    file_fs, waveform = wavfile.read(filename, mmap=False)
    # Rescale to range -1.0 to 1.0
    if waveform.dtype != np.float32:
        ii = np.iinfo(waveform.dtype)
        waveform = waveform.astype(np.float32)
        waveform = (waveform - ii.min) / (ii.max - ii.min) * 2 - 1

    if force_duration is not None:
        final_samples = int(force_duration*fs)
        if len(waveform) > final_samples:
            waveform = waveform[:final_samples]
            log.info(f'truncated to {final_samples} samples')
        elif len(waveform) < final_samples:
            waveform = np.concatenate([waveform, np.zeros(final_samples-len(waveform))])
            log.info(f'padded with {final_samples-len(waveform)} samples')

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
        waveform *= (1.4141 * sf)  # 1.4141 V RMS = 80 dB
        #log.info(f"Atten: {attenuatedB} SF: {sf} RMS: {waveform.std()}")

    waveform[waveform>5]=5
    waveform[waveform<-5]=-5
    #if np.max(np.abs(waveform)) > 5:
    #    raise ValueError('waveform value too large')

    # Resample if sampling rate does not match
    if fs != file_fs:
        waveform = util.resample_fft(waveform, file_fs, fs)

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

    def __init__(self, filenames, filelabels='', normalization='pe', level=65,
                 norm_fixed_scale=1, force_duration=None, random_seed=0, **kwargs):
        """
        :param filenames: list filenames (single channel) or list of lists
           (each file in inner list fed to a different channel)
        :param filelabels: list string labeling user-defined category for each filename
           (if you want to distinguish between target and catch, eg)
           if None, defaults to a list of empty strings
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
        if type(filelabels) is str:
            self.filelabels = [filelabels] * len(filenames)
        else:
            self.filelabels = filelabels

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
        all_wav = list(sorted(Path(path).glob('*.wav')))
        if len(all_wav)==0:
            log.info(f'No wav files found in path={path}')
        if duration > 0:
            force_duration = duration
        else:
            force_duration = None
        if test_range is None:
            test_range=slice(0)
        if fit_range is None:
            fit_range=slice(0)
        elif fit_range==-1:
            fit_range = slice(len(all_wav))

        #log.info("**************************************************")
        #log.info(path)
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

    TODO:
        1. Merge fg/bgset settings into main class


    """
    default_parameters = [
        {'name': 'fg_path', 'label': 'FG path', 'default': 'h:/sounds/vocalizations/v4', 'dtype': 'str'},
        {'name': 'bg_path', 'label': 'BG path', 'default': 'h:/sounds/backgrounds/v3', 'dtype': 'str'},
        {'name': 'fg_range', 'label': 'FG wav indexes', 'expression': '[0]'},
        {'name': 'bg_range', 'label': 'BG wav indexes', 'expression': '[0]'},

        {'name': 'normalization', 'label': 'Normalization', 'default': 'rms', 'type': 'EnumParameter',
         'choices': {'max': "'pe'", 'RMS': "'rms'", 'fixed': "'fixed'"}},
        {'name': 'norm_fixed_scale', 'label': 'fixed norm value', 'default': 1, 'dtype': float},
        {'name': 'fg_level', 'label': 'FG level(s) dB SNR', 'expression': '[55]', 'dtype': object},
        {'name': 'bg_level', 'label': 'BG level(s) dB SNR', 'expression': '[55]', 'dtype': object},
        {'name': 'duration', 'label': 'FG/BG duration (s)', 'default': 3.0, 'dtype': float},
        {'name': 'fg_delay', 'label': 'FG delay (s)', 'default': 0.0, 'dtype': float},

        {'name': 'primary_channel', 'label': 'Primary FG channel', 'default': 0, 'dtype': int},
        {'name': 'fg_switch_channels', 'label': 'Switch FG channel', 'type': 'BoolParameter', 'default': False, 'dtype': bool},
        {'name': 'combinations', 'label': 'How to combine FG+BG', 'default': 'all', 'type': 'EnumParameter',
         'choices': {'simple': "'simple'", 'all': "'all'"}},

        {'name': 'contra_n', 'label': 'Contra BG portion (int)', 'default': 1, 'dtype': int},
        {'name': 'diotic_n', 'label': 'Diotic BG portion (int)', 'default': 0, 'dtype': int},
        {'name': 'ipsi_n', 'label': 'Ipsi BG portion (int)', 'default': 0, 'dtype': int},

        {'name': 'migrate_fraction', 'label': 'Percent migrate trials', 'default': 0.0, 'type': 'EnumParameter',
         'choices': {'0': 0.0, '25': 0.25, '50': 0.5}},
        {'name': 'migrate_start', 'label': "migrate_start (s)", 'default': 0.5, 'dtype': float},
        {'name': 'migrate_stop', 'label': "migrate_stop (s)", 'default': 1.0, 'dtype': float},

        {'name': 'response_window', 'label': 'Response start,stop (s)', 'expression': '(0, 1)'},
        {'name': 'reward_ambiguous_frac', 'label': 'Frac. reward ambiguous', 'default': 1.0, 'type': 'EnumParameter',
         'choices': {'all': 1.0, 'random 50%': 0.5, 'never': 0.0}},

        {'name': 'random_seed', 'label': 'Random seed', 'default': 0, 'dtype': int},
        {'name': 'fs', 'label': 'Sampling rate (sec^-1)', 'default': 44000, },

        {'name': 'fg_channel', 'label': 'FG chan', 'type': 'Result'},
        {'name': 'bg_channel', 'label':  'BG chan', 'type': 'Result'},
        {'name': 'fg_name', 'label':  'FG', 'type': 'Result'},
        {'name': 'bg_name', 'label':  'BG', 'type': 'Result'},
        {'name': 'this_snr', 'label':  'Trial SNR', 'type': 'Result'},
        {'name': 'migrate_trial', 'label':  'Moving Tar', 'type': 'Result'},
    ]

    for d in default_parameters:
        # Use `setdefault` so we don't accidentally override a parameter that
        # wants to use a different group.
        d.setdefault('group_name', 'FgBgSet')

    @classmethod
    def default_values(self):
        '''
        Returns a list of default values for each parameter for testing.

        The actual values of the parameters are evaluated by `psi.context`, but
        we want to be able to set these ourselves in test scripts.
        '''
        values = {}
        for d in self.default_parameters:
            if d.get('type', '') == 'Result':
                pass  # used for psi display and logging
            elif 'expression' in d.keys():
                values[d['name']] = eval(d['expression'])
                setattr(self, d['name'], eval(d['expression']))
            else:
                values[d['name']] = d['default']
        return values

    def __init__(self, **parameter_dict):
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
        # trial management
        self.current_trial_idx = -1
        self.trial_wav_idx = np.array([], dtype=int)
        self.trial_outcomes = np.array([], dtype=int)
        self.trial_is_repeat = np.array([], dtype=int)
        self.current_full_rep = 0
        self.fg_snr = 0

        self.update_parameters(parameter_dict)

    @property
    def user_parameters(self):
        return self.default_parameters

    def update_parameters(self, parameter_dict):
        for k, v in parameter_dict.items():
            setattr(self, k, v)

        self.FgSet = MCWavFileSet(
            fs=self.fs, path=self.fg_path, duration=self.duration,
            normalization=self.normalization, fit_range=self.fg_range,
            test_range=slice(0, ), test_reps=1, channel_count=1, level=65)
        self.BgSet = MCWavFileSet(
            fs=self.fs, path=self.bg_path, duration=self.duration,
            normalization=self.normalization, fit_range=self.bg_range,
            test_range=slice(0, ), test_reps=1, channel_count=1, level=65)
        self.update()

    @property
    def wav_per_rep(self):
        return np.min([len(self.bg_index), len(self.fg_index)])

    def update(self, trial_idx=None):
        """figure out indexing to map wav_set idx to specific members of FgSet and BgSet.
        manage trials separately to allow for repeats, etc."""
        _rng = np.random.RandomState(self.random_seed)

        if self.combinations == 'simple':
            bg_range = list(np.arange(self.BgSet.max_index, dtype=int))
            fg_range = list(np.arange(self.FgSet.max_index, dtype=int))

            if len(fg_range)>len(bg_range):
                while len(bg_range)<len(fg_range):
                    bg_range += bg_range
                bg_range=bg_range[:len(fg_range)]
            elif len(fg_range)<len(bg_range):
                while len(fg_range)<len(bg_range):
                    fg_range += fg_range
                fg_range=fg_range[:len(bg_range)]
        elif self.combinations == 'all':
            bg_range_ = list(np.arange(self.BgSet.max_index, dtype=int))
            fg_range_ = list(np.arange(self.FgSet.max_index, dtype=int))

            fg_range = fg_range_ * len(bg_range_)
            bg_range=[]
            [bg_range.extend([b]*len(fg_range_)) for b in bg_range_];
        data = {'fg_index': fg_range, 'bg_index': bg_range, 'fg_go': 1, 'fg_delay': self.fg_delay}
        log.info(f"{data}")
        stim = pd.DataFrame(data={'fg_index': fg_range, 'bg_index': bg_range, 'fg_go': 1, 'fg_delay': self.fg_delay},
                            columns=['fg_index', 'bg_index', 'fg_channel', 'bg_channel', 'fg_level', 'bg_level',
                                     'fg_go', 'fg_delay', 'migrate_trial'])

        bg_channels = [self.primary_channel] * self.ipsi_n + \
                      [1-self.primary_channel] * self.contra_n + \
            [-1] * self.diotic_n
        fg_channels = [self.primary_channel] * len(bg_channels)
        if self.fg_switch_channels:
            bg_channels += [1-self.primary_channel] * self.ipsi_n + \
                           [self.primary_channel] * self.contra_n + \
                           [-1] * self.diotic_n
            fg_channels += [1-self.primary_channel] * (self.ipsi_n + self.contra_n + self.diotic_n)

        dlist = []
        for f,b in zip(fg_channels, bg_channels):
            s = stim.copy()
            s['fg_channel']=f
            s['bg_channel']=b
            dlist.append(s)
        stim = pd.concat(dlist, ignore_index=True)

        if type(self.fg_level) is int:
            stim['fg_level'] = self.fg_level
        else:
            dlist = []
            for f in self.fg_level:
                s = stim.copy()
                s['fg_level'] = f
                dlist.append(s)
            stim = pd.concat(dlist, ignore_index=True)
        if type(self.bg_level) is int:
            stim['bg_level'] = self.bg_level
        else:
            dlist = []
            for f in self.bg_level:
                s = stim.copy()
                s['bg_level'] = f
                dlist.append(s)
            stim = pd.concat(dlist, ignore_index=True)
        stim = stim.loc[(stim['fg_level']>0) | (stim['bg_level']>0)]

        # remove dups of zero-dB spatial locations
        # but allow other dups
        zrows = (stim['bg_level']==0) | (stim['fg_level']==0)
        nzrows = (stim['bg_level']>0) & (stim['fg_level']>0)
        zstim = stim.loc[zrows].copy()
        nzstim = stim.loc[nzrows].copy()
        zstim.loc[zstim['fg_level']==0, 'fg_channel']=-1
        zstim.loc[zstim['fg_level']==0, 'fg_index']=stim['fg_index'].min()
        zstim.loc[zstim['bg_level']==0, 'bg_channel']=-1
        zstim.loc[zstim['bg_level']==0, 'bg_index']=stim['bg_index'].min()
        stim = pd.concat([nzstim, zstim.drop_duplicates()], ignore_index=True)

        # check if any stims are labeled catch, and set fg_go accordingly:
        for b in set(bg_range):
            if self.BgSet.filelabels[b] == 'C':
                stim.loc[stim['bg_index']==b,'fg_go']=-1
        for f in set(fg_range):
            if self.FgSet.filelabels[f] == 'C':
                stim.loc[stim['fg_index'] == f, 'fg_go'] = -1
        stim.loc[stim['fg_level']==0, 'fg_go'] = -1

        if self.migrate_fraction>=1:
            migrate_list = [1]
        elif (self.migrate_fraction > 0.4):
            migrate_list = [0, 1]
        elif (self.migrate_fraction > 0.3):
            migrate_list = [0, 0, 1]
        elif (self.migrate_fraction > 0.0):
            migrate_list = [0, 0, 0, 1]
        else:
            migrate_list = [0]

        dlist = []
        for migrate_trial in migrate_list:
            s = stim.copy()
            s['migrate_trial'] = bool(migrate_trial)
            dlist.append(s)
        stim = pd.concat(dlist, ignore_index=True)

        migrate_keep = (stim['fg_level'] > 0) | (stim['migrate_trial'] == False)
        stim = stim.loc[migrate_keep]

        self.stim_list = stim.reset_index()

        total_wav_set = len(stim)

        #if (type(self._fg_delay) is np.array) | (type(self._fg_delay) is list):
        #    self.fg_delay = np.array(self._fg_delay)
        #else:
        #    self.fg_delay = np.zeros(self.FgSet.max_index) + self._fg_delay

        # set up wav_set_idx to trial_idx mapping  -- self.trial_wav_idx
        if trial_idx is None:
            trial_idx = self.current_trial_idx

        if trial_idx >= len(self.trial_wav_idx):
            dd = 10
            ii = 0
            new_trial_wav = _rng.permutation(np.arange(total_wav_set, dtype=int))
            while (dd > 3) & (ii < 10) & (len(self.stim_list) > 3):
                new_trial_wav = _rng.permutation(np.arange(total_wav_set, dtype=int))
                dd = np.argwhere(np.diff(self.stim_list.loc[new_trial_wav,'fg_channel'])!=0).min()
                ii += 1
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

        row = self.stim_list.loc[wav_set_idx]

        wfg = self.FgSet.waveform(row['fg_index'])
        if row['fg_channel'] == 1:
            wfg = np.concatenate((np.zeros_like(wfg), wfg), axis=1)
        wbg = self.BgSet.waveform(row['bg_index'])
        if row['bg_channel'] == 1:
            wbg = np.concatenate((np.zeros_like(wbg), wbg), axis=1)
        elif row['bg_channel'] == -1:
            wbg = np.concatenate((wbg/(2**0.5), wbg/(2**0.5)), axis=1)


        fg_level = row['fg_level']
        bg_level = row['bg_level']
        if fg_level==0:
            fg_scaleby=0
        else:
            fg_scaleby = 10**((fg_level - self.FgSet.level)/20)
        if bg_level==0:
            bg_scaleby=0
        else:
            bg_scaleby = 10**((bg_level - self.BgSet.level)/20)
        wfg *= fg_scaleby
        wbg *= bg_scaleby

        log.info(f"fg level: {fg_level} bg level: {bg_level} FG RMS: {wfg.std():.3f} BG RMS: {wbg.std():.3f}")


        if wbg.shape[1] < 2:
            wbg = np.concatenate((wbg, np.zeros_like(wbg)), axis=1)
        if wfg.shape[1] < 2:
            wfg = np.concatenate((wfg, np.zeros_like(wfg)), axis=1)

        if row['migrate_trial']:
            log.info(f'This is a target migration trial {self.migrate_start}->{self.migrate_stop} s')
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
        offsetbins = int(row['fg_delay'] * self.FgSet.fs)
        w = wbg
        if wfg.shape[0]+offsetbins > wbg.shape[0]:
            print(wfg.shape[0], offsetbins, wbg.shape[0])
            w = np.concatenate((w, np.zeros((wfg.shape[0]+offsetbins-wbg.shape[0],
                                             wbg.shape[1]))), axis=0)
        w[offsetbins:(offsetbins+wfg.shape[0]), :] += wfg

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
        row = self.stim_list.loc[wav_set_idx]

        fg_i = row['fg_index']
        bg_i = row['bg_index']

        is_go_trial = row['fg_go']
        if is_go_trial==-1:
            # -1 means either port
            if (self.reward_ambiguous_frac==0.5):
                response_condition = int(np.ceil(np.random.uniform(0, 2)))
            elif (self.reward_ambiguous_frac==0):
                response_condition = 0
            else:
                response_condition = -1
        elif is_go_trial==1:
            # 1=spout 1, 2=spout 2
            response_condition = int(row['fg_channel']+1)
        else:
            response_condition = 0

        if type(self.response_window) is tuple:
            response_window = (row['fg_delay'] + self.response_window[0],
                               row['fg_delay'] + self.response_window[1])
        else:
            response_window = (row['fg_delay'] + self.response_window[fg_i][0],
                               row['fg_delay'] + self.response_window[fg_i][1])

        d = {'trial_idx': trial_idx,
             'wav_set_idx': wav_set_idx,
             'fg_i': fg_i,
             'bg_i': bg_i,
             'fg_name': self.FgSet.names[fg_i],
             'bg_name': self.BgSet.names[bg_i],
             'fg_duration': self.FgSet.duration,
             'bg_duration': self.BgSet.duration,
             'snr': row['fg_level']-row['bg_level'],
             'fg_level': row['fg_level'],
             'bg_level': row['bg_level'],
             'this_snr': row['fg_level']-row['bg_level'],
             'fg_delay': row['fg_delay'],
             'fg_channel': row['fg_channel'],
             'bg_channel': row['bg_channel'],
             'migrate_trial': row['migrate_trial'],
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


class FgBgSet_old(WavSet):
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
                 random_seed=0, reward_ambiguous_frac=1):
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
        self.reward_ambiguous_frac = reward_ambiguous_frac

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

        #region set fg an bg
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
        fgcmax = fgc.max()

        bgimin = bgi.min()
        bgcmax = bgc.max()
        snr_keep = ((fsnr<=50) | ((bgi==bgimin) & (bgc==bgcmax))) & \
                   ((fsnr>-100) | ((fgi==fgimin) & (fgc==fgcmax)))
        bgi = bgi[snr_keep]
        fgi = fgi[snr_keep]
        bgc = bgc[snr_keep]
        fgc = fgc[snr_keep]
        fsnr = fsnr[snr_keep]

        # fgg provides information about whether this is a
        # go (>0)/no-go (0)/go-anywhere (-1) trial
        fgg = fgg[snr_keep]
        fgg[fsnr<=-100]=-1

        # check if any stims are labeled catch:
        for i, b in enumerate(bgi):
            if self.BgSet.filelabels[b] == 'C':
                fgg[i]=-1
        for i, f in enumerate(fgi):
            if self.FgSet.filelabels[f] == 'C':
                fgg[i]=-1
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

        # Important
        self.bg_index = bgi
        self.fg_index = fgi
        self.fg_channel = fgc
        self.bg_channel = bgc
        self.fg_snr = fsnr
        self.fg_go = fgg
        self.migrate_trial = migrate_trial
        # ------- Important

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
            if (self.reward_ambiguous_frac==0.5):
                response_condition = int(np.ceil(np.random.uniform(0,2)))
            elif (self.reward_ambiguous_frac==0):
                response_condition = 0
            else:
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
        if s1idx>-1:
            s1_name = self.wavset.names[s1idx]
        else:
            s1_name = ''
        if s2idx>-1:
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

#
class CategorySet(FgBgSet):
    """
    CategorySet class: modified from the FgBgSet class
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

    Miscellaneous comments:
        Foreground = Target.
        Background = Non-target.

    """

    def __init__(self, FgSet=None, BgSet=None, CatchFgSet=None, CatchBgSet=None, OAnoiseSet=None,
                 combinations='custom',fg_switch_channels=True, bg_switch_channels=False, primary_channel=0,
                 fg_delay=0.0, fg_snr=0.0, response_window=None, random_seed=0, catch_ferret_id=4,
                 n_env_bands=[2, 8, 32], reg2catch_ratio=6, unique_overall_SNR= [np.inf]):
        """
        FgBgSet polls FgSet and BgSet for .max_index, .waveform, .names, and .fs
        :param FgSet: {WaveformSet, MultichannelWaveformSet, None}
        :param BgSet: {WaveformSet, MultichannelWaveformSet, None}
        :param CatchBgSet: {WaveformSet, MultichannelWaveformSet, None}
        :param combinations: str
            simple: random pairing
            all: exhaustive, every possible combination of bg and fg
            custom: for user-defined case
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
        if CatchFgSet is None:
            if FgSet is None:
                self.FgSet = WaveformSet()
            else:
                self.FgSet = FgSet
        else:
            if FgSet is None:
                raise ValueError(f"Must specify FgSet if CatchFgSet is specified")
            else:
                self.FgSet = cat_MCWavFileSets(FgSet, CatchFgSet)

        if CatchBgSet is None:
            if BgSet is None:
                self.BgSet = WaveformSet()
            else:
                self.BgSet = BgSet
        else:
            if BgSet is None:
                raise ValueError(f"Must specify BgSet if CatchBgSet is specified")
            else:
                self.BgSet = cat_MCWavFileSets(BgSet, CatchBgSet)

        if OAnoiseSet is None:
            self.OAnoiseSet = WaveformSet()
            if not all(np.isinf(unique_overall_SNR)):
                raise ValueError(f"unique_overall_SNR must be [np.inf] if OAnoiseSet is None")
        else:
            self.OAnoiseSet = OAnoiseSet

        self.combinations = combinations
        self.fg_switch_channels = fg_switch_channels
        self.bg_switch_channels = bg_switch_channels
        # self.catch_frequency = catch_frequency
        self._fg_delay = fg_delay
        self._fg_snr = fg_snr
        self.primary_channel = primary_channel

        self.catch_ferret_id = catch_ferret_id
        self.n_env_bands = n_env_bands
        self.reg2catch_ratio = reg2catch_ratio

        self.unique_overall_SNR = unique_overall_SNR

        # #region migrate code
        # self.migrate_fraction = migrate_fraction
        # self.migrate_start = migrate_start
        # self.migrate_stop = migrate_stop
        # #endregion

        assert combinations == 'custom', 'Only set up for user-defined combinations'

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

    # inherit from FgBgSet
    def update(self, trial_idx=None):
        """figure out indexing to map wav_set idx to specific members of FgSet and BgSet.
        manage trials separately to allow for repeats, etc."""
        _rng = np.random.RandomState(self.random_seed)
        np.random.seed(self.random_seed)

        if all(np.isinf(self.unique_overall_SNR)):
            fgi, bgi, fgg = get_stim_list(self.FgSet, self.BgSet, self.catch_ferret_id, self.n_env_bands, self.reg2catch_ratio)
            overall_snr = np.inf * np.ones_like(fgi)
        else:
            fgi, bgi, fgg = get_stim_list_no_catch(self.FgSet, self.BgSet)
            # add np.inf as the first snr if doesn't already exist
            self.unique_overall_SNR = [np.inf] + list(set(self.unique_overall_SNR) - set([np.inf]))
            num_noninf_snrs = len(self.unique_overall_SNR) - 1
            overall_snr_weight = np.concatenate(([num_noninf_snrs], np.ones(num_noninf_snrs)))
            overall_snr_prob = overall_snr_weight / overall_snr_weight.sum()
            # overall_snr = choices(self.unique_overall_SNR, weights = overall_snr_weight, k = len(fgi))
            overall_snr = np.random.choice(self.unique_overall_SNR, p=overall_snr_prob, size=len(fgi))
        num_uniq_trial = len(fgi)

        # region set fg an bg
        if self.fg_switch_channels:
            fgc = np.concatenate((np.zeros_like(fgi), np.ones_like(fgi)))
            fgi, bgi, fgg, overall_snr = np.tile(fgi, 2), np.tile(bgi, 2), np.tile(fgg, 2), np.tile(overall_snr, 2)
            num_uniq_trial *= 2
        else:
            fg_channel = np.zeros_like(fgi) + self.primary_channel

        if self.bg_switch_channels == False:
            bgc = np.zeros_like(fgi)
        elif self.bg_switch_channels == 'same':
            bgc = fgc.copy()
        elif self.bg_switch_channels == 'opposite':
            bgc = 1 - fgc
        else:
            raise ValueError(f"Unknown bg_switch_channels value: {self.bg_switch_channels}.")

        print(f"{self._fg_snr}: type = {type(self._fg_snr)}")

        assert len(self._fg_snr)==1, "Only a single SNR is supported"
        fsnr = np.zeros_like(fgi) + self._fg_snr
        migrate_trial = np.zeros_like(fgi)

        total_wav_set = len(fgg)

        # Important
        self.bg_index = bgi
        self.fg_index = fgi
        self.fg_channel = fgc
        self.bg_channel = bgc
        self.fg_snr = fsnr
        self.fg_go = fgg
        self.migrate_trial = migrate_trial
        self.overall_snr = overall_snr
        # ------- Important

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
        log.info(f"wfg={wfg.shape},wbg={wbg.shape}, bfdur={self.BgSet.duration},fg level: {self.FgSet.level} bg level: {self.BgSet.level} FG RMS: {wfg.std():.3f} BG RMS: {wbg.std():.3f}")

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
        log.info(f"flag::bg::: wbg={wbg.shape},offsetbins={offsetbins},fg_delay={self.fg_delay}")

        # combine fg and bg waveforms
        w = wbg
        if wfg.shape[0]+offsetbins > wbg.shape[0]:
            log.info(f"'<qqqqqqq>', {wfg.shape[0]}, {offsetbins}, {wbg.shape[0]}")
            w = np.concatenate((w, np.zeros((wfg.shape[0]+offsetbins-wbg.shape[0],
                                             wbg.shape[1]))), axis=0)
        w[offsetbins:(offsetbins+wfg.shape[0]), :] += wfg * fg_scale
        if w.shape[1] < 2:
            w = np.concatenate((w, np.zeros_like(w)), axis=1)

        if not np.isinf(self.overall_snr[wav_set_idx]):
            cur_overall_snr = self.overall_snr[wav_set_idx]
            overall_noise_scale = 10 ** (-cur_overall_snr / 20)
            # noise and ferret vocal can be matched in index
            ov_noise = self.OAnoiseSet.waveform(self.fg_index[wav_set_idx])
            ov_noise = np.concatenate((ov_noise,ov_noise), axis=1)
            sig_len = w.shape[0]
            ov_noise = ov_noise[:sig_len, :]
            log.info(f"<flag><><><><><><>< {trial_idx}|{wav_set_idx}| snr = {cur_overall_snr}: w={w.shape}, ov_noise{ov_noise.shape}")
            w = w/overall_noise_scale + ov_noise # will be ~3 dB louder than clean

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
             'overall_snr': self.overall_snr[wav_set_idx],
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
        #def score_response(self, outcome, repeat_incorrect=2, trial_idx=None):


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


