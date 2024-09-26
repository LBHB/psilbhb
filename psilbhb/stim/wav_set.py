from functools import partial, lru_cache
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
    be_verbose = 0
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

    # catch_fg_names = [soundpath_fg + '/' + catch_fgs[idx] for idx in catch_fg_inds]
    catch_bg_names = [soundpath_catch_bg + '/' + 'ENV{}_{}'.format(nb,catch_fgs[bg_idx])
                      for bg_idx,nb in zip(catch_bg_inds,catch_env_nbands)]

    # if be_verbose:
    #     print("~~~~~~~~~Catch~~~~~~~~~")
    #     [print(catch_fg_names[i] + ' vs ' + catch_bg_names[i]) for i in range(num_catch_trials)]
    #     print("~~~~~~~~~end Catch~~~~~~~~~")
    #endregion

    #region get all regular (non-catch) trial stimuli: fgs and speech
    num_regular_trials = reg2catch_ratio*num_catch_trials

    # taboo_ferret_ids = [1, 2, 7, catch_ferret_id]
    taboo_ferret_ids = [1, 2, 7]
    # reg_fg_names = choices([soundpath_fg + '/' + x for x in all_ferret_files
    #                         if not any([x.startswith("ferretb{}".format(tabid)) for tabid in taboo_ferret_ids])],
    #                        k=num_regular_trials)
    reg_fg_names = list(np.random.choice([soundpath_fg + '/' + x for x in all_ferret_files
                                     if not any([x.startswith("ferretb{}".format(tabid)) for tabid in taboo_ferret_ids])],
                                    size=num_regular_trials+num_catch_trials))

    # reg_bg_names = [soundpath_bg + '/' + x for x in choices(all_speech_files, k=num_regular_trials)]
    reg_bg_names = [soundpath_bg + '/' + x for x in np.random.choice(all_speech_files, size=num_regular_trials)]
    if be_verbose:
        print("~~~~~~~~~Regular~~~~~~~~~")
        [print(reg_fg_names[i] + ' vs ' + reg_bg_names[i]) for i in range(num_regular_trials)]
        print("~~~~~~~~~end Regular~~~~~~~~~")
    #endregion

    session_fg_files = reg_fg_names # + catch_fg_names
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
    num_regular_trials = 100
    # num_regular_trials = 10

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
    fgg = np.ones(num_regular_trials).astype(int)

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
    out_assert = np.array([getattr(set1, _atb) == getattr(set2, _atb) for _atb in compatibility_fields])
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

    # Resample if sampling rate does not match
    if fs != file_fs:
        waveform = util.resample_fft(waveform, file_fs, fs)

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
        # 3.5349 V RMS = 80 dB tone
        waveform = remove_clicks(waveform / util.rms(waveform), max_threshold=15) * 3.5349
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
        waveform *= sf
        #log.info(f"Atten: {attenuatedB} SF: {sf} RMS: {waveform.std()}")

    waveform[waveform>5]=5
    waveform[waveform<-5]=-5
    #if np.max(np.abs(waveform)) > 5:
    #    raise ValueError('waveform value too large')


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

    default_parameters = [
        {'name': 'fg_path', 'label': 'FG path', 'default': 'h:/sounds/vocalizations/v4', 'dtype': 'str'},
    ]

    def __init__(self, n_response):
        self.n_response = n_response
        self.current_trial_idx = 0
        self.trial_wav_idx = np.array([], dtype=int)
        self.trial_outcomes = np.array([], dtype=int)
        self.trial_is_repeat = np.array([], dtype=int)
        self.current_full_rep = 0

        self.stim_list = pd.DataFrame()

    @property
    def wav_per_rep(self):
        return len(self.stim_list)

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

    def stim_row(self, trial_idx=None, wav_set_idx=None):
        if trial_idx == 0:
            raise ValueError('trial_idx=0 not valid, starts counting at 1')

        if wav_set_idx is None:
            if trial_idx is None:
                self.current_trial_idx += 1
                trial_idx = self.current_trial_idx
            if len(self.trial_wav_idx) <= trial_idx:
                self.update(trial_idx=trial_idx)

            wav_set_idx = self.trial_wav_idx[trial_idx-1]
            #print(f"trial_idx={trial_idx} wav_set_idx={wav_set_idx}")

        if wav_set_idx < len(self.stim_list):
            r = self.stim_list.loc[[wav_set_idx]]
            # for i, row in r.iterrows():
            #     pass
            for row_tup in r.itertuples(index=False):
                row = row_tup._asdict()  # Convert named tuple to dictionary
                print(row)

        else:
            row = pd.Series({'index': wav_set_idx})

        return row

    @property
    def user_parameters(self):
        return self.default_parameters

    def update_parameters(self, parameter_dict):
        for k, v in parameter_dict.items():
            setattr(self, k, v)
        self.update()

    def update(self):
        pass

    def trial_waveform(self, trial_idx=None, wav_set_idx=None):
        pass

    def score_response(self, outcome, repeat_incorrect=1, trial_idx=None):
        """
        current logic: if invalid or incorrect, trial should be repeated
        :param outcome: int
            -1 trial not scored (yet?) - happens if score_response skips a trial_idx
            0 invalid
            1 incorrect
            2 correct
            3 correct - either response ok
        :param repeat_incorrect: int
            choices = {'No': 0, 'Early only': 1, 'Yes': 2}
            0: never repeat
            1: repeat if outcome = 0
            2: repeat if outcome in [0,1]
        :param trial_idx: int
            Values are decremented by 1 to match pythonic 0-based indexing.
            Must be >0 and <len(trial_wav_idx) to be valid. By default, function updates
            score for current_trial_idx and increments current_trial_idx by 1.
        :return:
        """
        outstr = ['Invalid', 'Incorrect', 'Correct']
        if trial_idx is None:
            trial_idx = self.current_trial_idx
            # Only incrementing current trial index if trial_idx is None. Do we always
            # want to do this???
            self.current_trial_idx = trial_idx + 1

        if trial_idx >= len(self.trial_wav_idx):
            raise ValueError(f"attempting to score response for trial_idx out of range")

        if trial_idx > len(self.trial_outcomes):
            n = trial_idx - len(self.trial_outcomes)
            self.trial_outcomes = np.concatenate((self.trial_outcomes, np.zeros(n, dtype=int)))

        self.trial_outcomes[trial_idx-1] = int(outcome)
        try:
            stim_cat = self.stim_cat
            log.info('Checking if probe trial')
            trial_wav_idx=self.trial_wav_idx[trial_idx - 1]
            log.info(f"{stim_cat[trial_wav_idx]}")
            if stim_cat[trial_wav_idx]=='C':
                force_no_repeat=True
                log.info('Yes, probe trial, forcing not repeat')
            else:
                force_no_repeat=False
        except:
            force_no_repeat=False
        if force_no_repeat:
            log.info(f'Trial {trial_idx} outcome {outstr[outcome]}: probe trial, force no repeat')
        elif ((repeat_incorrect == 2) and (outcome in [0, 1])) or \
                ((repeat_incorrect == 1) and (outcome == 0)):
            log.info(f'Trial {trial_idx} outcome {outstr[outcome]}: repeating immediately')
            self.trial_wav_idx = np.concatenate((self.trial_wav_idx[:trial_idx],
                                                 [self.trial_wav_idx[trial_idx-1]],
                                                 self.trial_wav_idx[trial_idx:]))
            self.trial_is_repeat = np.concatenate((self.trial_is_repeat[:(trial_idx)],
                                                 [1],
                                                 self.trial_is_repeat[(trial_idx):]))
        else:
            log.info(f'Trial {trial_idx} outcome {outstr[outcome]}: moving on')


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
        {'name': 'norm_fixed_scale', 'label': 'fixed norm value', 'default': 1, 'dtype': 'float'},
        {'name': 'fg_level', 'label': 'FG level(s) dB SNR', 'expression': '[55]', 'dtype': 'object'},
        {'name': 'bg_level', 'label': 'BG level(s) dB SNR', 'expression': '[55]', 'dtype': 'object'},
        {'name': 'duration', 'label': 'FG/BG duration (s)', 'default': 3.0, 'dtype': 'float'},
        {'name': 'fg_delay', 'label': 'FG delay (s)', 'default': 0.0, 'dtype': 'float'},

        {'name': 'primary_channel', 'label': 'Primary FG channel', 'default': 0, 'dtype': 'int'},
        {'name': 'fg_switch_channels', 'label': 'Switch FG channel', 'type': 'BoolParameter', 'default': False},
        {'name': 'combinations', 'label': 'How to combine FG+BG', 'default': 'all', 'type': 'EnumParameter',
         'choices': {'simple': "'simple'", 'all': "'all'"}},

        {'name': 'fg_choice_trials', 'label': 'FG choice portion (int)', 'default': 0, 'dtype': 'int'},
        {'name': 'contra_n', 'label': 'Contra BG portion (int)', 'default': 1, 'dtype': 'int'},
        {'name': 'diotic_n', 'label': 'Diotic BG portion (int)', 'default': 0, 'dtype': 'int'},
        {'name': 'ipsi_n', 'label': 'Ipsi BG portion (int)', 'default': 0, 'dtype': 'int'},

        {'name': 'migrate_fraction', 'label': 'Percent migrate trials', 'default': '0', 'type': 'EnumParameter',
         'choices': {'0': 0.0, '25': 0.25, '50': 0.5}},
        {'name': 'migrate_start', 'label': "migrate_start (s)", 'default': 0.5, 'dtype': 'float'},
        {'name': 'migrate_stop', 'label': "migrate_stop (s)", 'default': 1.0, 'dtype': 'float'},

        {'name': 'response_window', 'label': 'Response start,stop (s)', 'expression': '(0, 1)'},
        {'name': 'reward_ambiguous_frac', 'label': 'Frac. reward ambiguous', 'default': 'all', 'type': 'EnumParameter',
         'choices': {'all': 1.0, 'random 50%': 0.5, 'never': 0.0}},
        {'name': 'reward_durations', 'label': 'FG reward durations', 'expression': '()'},

        {'name': 'random_seed', 'label': 'Random seed', 'default': 0, 'dtype': 'int'},
        {'name': 'fs', 'label': 'Sampling rate (sec^-1)', 'default': 44000, 'group_name': 'Results'},

        {'name': 'fg_channel', 'label': 'FG chan', 'type': 'Result', 'group_name': 'Results'},
        {'name': 'bg_channel', 'label': 'BG chan', 'type': 'Result', 'group_name': 'Results'},
        {'name': 'fg_name', 'label': 'FG', 'type': 'Result', 'group_name': 'Results'},
        {'name': 'bg_name', 'label': 'BG', 'type': 'Result', 'group_name': 'Results'},
        {'name': 'this_snr', 'label': 'Trial SNR', 'type': 'Result', 'group_name': 'Results'},
        {'name': 'migrate_trial', 'label': 'Moving Tar', 'type': 'Result', 'group_name': 'Results'},
        {'name': 'current_full_rep', 'label': 'Rep', 'type': 'Result', 'group_name': 'Results'},
        {'name': 'trial_cat', 'label': 'Type', 'type': 'Result', 'group_name': 'Results'},
    ]

    for d in default_parameters:
        # Use `setdefault` so we don't accidentally override a parameter that
        # wants to use a different group.
        d.setdefault('group_name', 'FgBgSet')

    def __init__(self, n_response, **parameter_dict):
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
        super().__init__(n_response=n_response)

        self.fg_snr = 0
        self.update_parameters(parameter_dict)

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

        go_trials = [1] * len(fg_range)

        data = {'fg_index': fg_range, 'bg_index': bg_range, 'fg_go': go_trials, 'fg_delay': self.fg_delay}
        log.info(f"{pd.DataFrame(data)}")
        stim = pd.DataFrame(data={'fg_index': fg_range, 'bg_index': bg_range, 'fg_go': go_trials,
                                  'fg_delay': self.fg_delay},
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

        if self.fg_choice_trials > 0:
            fgc_range = [0] * self.fg_choice_trials * 2
            bgc_range = [1] * self.fg_choice_trials * 2
            fgc_channels = [0] * self.fg_choice_trials + [1] * self.fg_choice_trials
            bgc_channels = [1] * self.fg_choice_trials + [0] * self.fg_choice_trials

            stimc = pd.DataFrame(data={'fg_index': fgc_range, 'bg_index': bgc_range,
                                       'fg_channel': fgc_channels, 'bg_channel': bgc_channels,
                                       'fg_go': -2, 'fg_delay': self.fg_delay},
                                columns=['fg_index', 'bg_index', 'fg_channel', 'bg_channel', 'fg_level', 'bg_level',
                                         'fg_go', 'fg_delay', 'migrate_trial'])
            dlist.append(stimc)

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
        stim = stim.loc[((stim['fg_level']>0) & (stim['fg_go']>-2)) |
                        ((stim['bg_level']>0) & (stim['fg_go']>-2)) |
                        ((stim['fg_level']>0) & (stim['bg_level']>0))]

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
        for b in set(stim.loc[(stim['fg_go'] == 1), 'bg_index'].values):
            if self.BgSet.filelabels[b] == 'C':
                stim.loc[(stim['bg_index'] == b) & (stim['fg_go'] == 1), 'fg_go'] = -1
        for f in set(fg_range):
            if self.FgSet.filelabels[f] == 'C':
                stim.loc[stim['fg_index'] == f, 'fg_go'] = -1
        stim.loc[stim['fg_level'] == 0, 'fg_go'] = -1

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
            # fix to prevent identical sequences from repeating
            for t in range(trial_idx):
                _ = _rng.permutation(np.arange(total_wav_set, dtype=int))
            new_trial_wav = _rng.permutation(np.arange(total_wav_set, dtype=int))
            while (dd > 3) & (ii < 10) & (len(self.stim_list) > 3):
                new_trial_wav = _rng.permutation(np.arange(total_wav_set, dtype=int))
                dd = np.argwhere(np.diff(self.stim_list.loc[new_trial_wav, 'fg_channel'])!=0).min()
                ii += 1
            self.trial_wav_idx = np.concatenate((self.trial_wav_idx, new_trial_wav))
            log.info(f'Added {len(new_trial_wav)}/{len(self.trial_wav_idx)} trials to trial_wav_idx')
            self.current_full_rep += 1
            self.trial_is_repeat = np.concatenate((self.trial_is_repeat, np.zeros_like(new_trial_wav)))

    def trial_waveform(self, trial_idx=None, wav_set_idx=None):
        row = self.stim_row(trial_idx=trial_idx, wav_set_idx=wav_set_idx)

        wfg = self.FgSet.waveform(row['fg_index'])
        if row['fg_channel'] == 1:
            wfg = np.concatenate((np.zeros_like(wfg), wfg), axis=1)
        if row['fg_go']==-2:
            # choice trial, FgSet for both channels
            wbg = self.FgSet.waveform(row['bg_index'])
        else:
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
        total_bins = int(self.duration*self.FgSet.fs)
        offsetbins = int(row['fg_delay'] * self.FgSet.fs)
        if wbg.shape[0]>total_bins:
            wbg=wbg[:total_bins]
        w = np.zeros((total_bins,wbg.shape[1]))
        w[:wbg.shape[0], :] = wbg

        if wfg.shape[0]+offsetbins > wbg.shape[0]:
            wfg = wfg[:(total_bins-offsetbins), :]
        w[offsetbins:(offsetbins+wfg.shape[0]), :] += wfg

        return w.T

    def trial_parameters(self, trial_idx=None, wav_set_idx=None):
        row = self.stim_row(trial_idx=trial_idx, wav_set_idx=wav_set_idx)

        fg_i = row['fg_index']
        bg_i = row['bg_index']

        is_go_trial = row['fg_go']
        if is_go_trial==-2:
            # choice trial - 2 fgs with different reward
            response_condition = -1

        elif is_go_trial == -1:
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

        if is_go_trial>=0:
            trial_cat='normal'
            fg_name = self.FgSet.names[fg_i]
            bg_name = self.BgSet.names[bg_i]
        elif is_go_trial == -1:
            trial_cat='catch'
            fg_name = 'null'
            bg_name = self.BgSet.names[bg_i]
        else:
            trial_cat='choice'
            fg_name = self.FgSet.names[fg_i]
            bg_name = self.FgSet.names[bg_i]

        d = {'trial_idx': trial_idx,
             'wav_set_idx': row['index'],
             'fg_i': fg_i,
             'bg_i': bg_i,
             'fg_name': fg_name,
             'bg_name': bg_name,
             'fg_duration': self.FgSet.duration,
             'bg_duration': self.BgSet.duration,
             'snr': row['fg_level']-row['bg_level'],
             'this_fg_level': row['fg_level'],
             'this_bg_level': row['bg_level'],
             'this_snr': row['fg_level']-row['bg_level'],
             'fg_delay': row['fg_delay'],
             'fg_channel': row['fg_channel'],
             'bg_channel': row['bg_channel'],
             'migrate_trial': row['migrate_trial'],
             'response_condition': response_condition,
             'response_window': response_window,
             'current_full_rep': self.current_full_rep,
             'primary_channel': self.primary_channel,
             'trial_is_repeat': self.trial_is_repeat[trial_idx-1],
             'trial_cat': trial_cat,
             }

        if (is_go_trial==-2) & (len(self.reward_durations)>1):
            # Override dispense durations
            d[f'dispense_duration'] = -1
            if row['fg_channel'] == 0:
                d[f'dispense_1_duration'] = self.reward_durations[fg_i]
                d[f'dispense_2_duration'] = self.reward_durations[bg_i]
            else:
                d[f'dispense_2_duration'] = self.reward_durations[fg_i]
                d[f'dispense_1_duration'] = self.reward_durations[bg_i]

        elif fg_i < len(self.reward_durations):
            # Override dispense durations
            d[f'dispense_duration'] = self.reward_durations[fg_i]
            d[f'dispense_1_duration'] = -1
            d[f'dispense_2_duration'] = -1

        # snippet from controller
        # for i in range(self.N_response):
        #    if 'dispense_{i+1}_duration' in wavset_info:
        #        self.context.set_value(f'water_dispense_{i+1}_duration', wavset_info[f'dispense_{i+1}_duration'])
        #    elif 'dispense_duration' in wavset_info:
        #        self.context.set_value(f'water_dispense_{i+1}_duration', wavset_info[f'dispense_duration'])

        return d

    def score_response_old(self, outcome, repeat_incorrect=2, trial_idx=None):
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

        if trial_idx >= len(self.trial_wav_idx):
            raise ValueError(f"attempting to score response for trial_idx out of range")

        if trial_idx>=len(self.trial_outcomes):
            n = trial_idx - len(self.trial_outcomes) + 1
            self.trial_outcomes = np.concatenate((self.trial_outcomes, np.zeros(n)-1))
        self.trial_outcomes[trial_idx] = int(outcome)

        wav_set_idx = self.trial_wav_idx[trial_idx]
        row = self.stim_list.loc[wav_set_idx]
        if (row['fg_go'] > -1) & (repeat_incorrect == 2 and (outcome in [0, 1])) \
                or (repeat_incorrect == 1 and (outcome in [0])):
            #log.info('Trial {trial_idx} outcome {outcome}: appending repeat to trial_wav_idx')
            #self.trial_wav_idx = np.concatenate((self.trial_wav_idx, [self.trial_wav_idx[trial_idx]]))
            #self.trial_is_repeat = np.concatenate((self.trial_is_repeat, [1]))
            # log.info('Trial {trial_idx} outcome {outcome}: appending repeat to trial_wav_idx')
            # self.trial_wav_idx = np.concatenate((self.trial_wav_idx, [self.trial_wav_idx[trial_idx]]))
            # self.trial_is_repeat = np.concatenate((self.trial_is_repeat, [1]))
            log.info(f'Trial {trial_idx} outcome {outcome}: repeating immediately')
            self.trial_wav_idx = np.concatenate((self.trial_wav_idx[:trial_idx],
                                                 [self.trial_wav_idx[trial_idx]],
                                                 self.trial_wav_idx[trial_idx:]))
            self.trial_is_repeat = np.concatenate((self.trial_is_repeat[:(trial_idx + 1)],
                                                   [1],
                                                   self.trial_is_repeat[(trial_idx + 1):]))
        elif row['fg_go']==-1:
            log.info(f'Trial {trial_idx} bg-only trial: moving on')
        else:
            log.info(f'Trial {trial_idx} outcome {outcome}: moving on')


class AMFusion(WavSet):

    default_parameters = [
        {'name': 'target_frequency', 'label': 'Target center frequenc(ies) (list)',
         'expression': '[1000]', 'dtype': 'object', 'scope': 'experiment'},
        {'name': 'target_am_rate', 'label': 'Target AM rate (list)',
         'expression': '[20]', 'dtype': 'object', 'scope': 'experiment'},
        {'name': 'target_bandwidth', 'label': 'Target bandwidth (0=PT)',
         'expression': '0', 'dtype': 'object', 'scope': 'experiment'},
        {'name': 'modulation_depth', 'label': 'Target modulation depth(s) (list)',
         'expression': '[100]', 'dtype': 'object', 'scope': 'experiment'},
        {'name': 'target_level', 'label': 'Target dB SPL (list)',
         'expression': '[60]', 'dtype': 'object', 'scope': 'experiment'},
        {'name': 'distractor_frequency', 'label': 'Distractor center frequenc(ies) (list)',
         'expression': '[4000]', 'dtype': 'object', 'scope': 'experiment'},
        {'name': 'distractor_level', 'label': 'Distractor dB SPL (list)',
         'expression': '[0]', 'dtype': 'object', 'scope': 'experiment'},
        {'name': 'duration', 'label': 'duration of each sample (s)',
         'default': 1.0, 'dtype': 'double', 'scope': 'experiment'},

        {'name': 'primary_channel', 'label': 'Primary channel',
         'compact_label': 'primary_channel', 'default': '0',
         'choices': {'0': 0, '1': 1},
         'scope': 'experiment', 'type': 'EnumParameter'},
        {'name': 'switch_channels', 'label': 'Switch target channel?',
         'compact_label': 'combinations', 'default': 'No',
         'choices': {'No': "False", 'Yes': "True"},
         'scope': 'experiment', 'type': 'EnumParameter'},
        {'name': 'reward_ambiguous_frac', 'label': 'Frac. reward ambiguous', 'default': 'all', 'type': 'EnumParameter',
         'choices': {'all': 1.0, 'random 50%': 0.5, 'never': 0.0}},

        {'name': 'fs', 'label': 'sampling rate (1/s)', 'default': 44000,
         'dtype': 'double', 'scope': 'experiment'},
        {'name': 'response_start', 'label': 'response win start (s)',
         'default': 0, 'dtype': 'double', 'scope': 'experiment'},
        {'name': 'response_end', 'label': 'response win end (s)', 'default': 2,
         'dtype': 'double', 'scope': 'experiment'},
        {'name': 'random_seed', 'label': 'random_seed', 'default': 0, 'dtype':
         'int', 'scope': 'experiment'},
        {'name': 'this_target_frequency', 'label': 'T', 'type': 'Result'},
        {'name': 'this_distractor_frequency', 'label': 'D', 'type': 'Result'},
        {'name': 'this_snr', 'label': 'SNR', 'type': 'Result'},
        {'name': 'response_condition', 'label': 'T spout', 'type': 'Result'},
        {'name': 'trial_is_repeat', 'label': 'rep', 'type': 'Result'},
    ]

    for d in default_parameters:
        # Use `setdefault` so we don't accidentally override a parameter that
        # wants to use a different group.
        d.setdefault('group_name', 'AMFusion')

    def __init__(self, n_response, **parameter_dict):
        super().__init__(n_response=n_response)
        # internal object to handle wavs, don't need to specify independently
        log.info('N_response %r', self.n_response)

        self.update_parameters(parameter_dict)


    def update_parameters(self, parameter_dict):
        for k, v in parameter_dict.items():
            setattr(self, k, v)
        self.response_window = (parameter_dict['response_start'], parameter_dict['response_end'])

        self.update()


    def update(self, trial_idx=None):
        """figure out indexing to map wav_set idx to specific members of FgSet and BgSet.
        manage trials separately to allow for repeats, etc."""
        _rng = np.random.RandomState(self.random_seed)

        tar_range_ = np.array(self.target_frequency, dtype=float)
        am_rate_ = np.array(self.target_am_rate, dtype=float)
        if len(am_rate_)==1:
            am_rate_ = np.zeros_like(tar_range_) + am_rate_
        am_depth_ = np.array(self.modulation_depth, dtype=float)
        if len(am_depth_)==1:
            am_depth_ = np.zeros_like(tar_range_) + am_depth_
        dis_range_ = np.array(self.distractor_frequency, dtype=float)

        # combinations
        tar_count=len(tar_range_)
        dis_count=len(dis_range_)
        slist = []
        for tlevel in self.target_level:
            for dlevel in self.distractor_level:
                data = {'tar_freq': np.concatenate([tar_range_] * dis_count),
                        'tar_am': np.concatenate([am_rate_] * dis_count),
                        'tar_depth': np.concatenate([am_depth_] * dis_count),
                        'tar_bandwidth': self.target_bandwidth,
                        'tar_level': tlevel,
                        'dis_freq': np.concatenate([np.zeros(tar_count)+d for d in dis_range_]),
                        'dis_level': dlevel,
                        'duration': self.duration,
                        'tar_channel': self.primary_channel,
                        }
                log.info(f"{data}")
                slist.append(pd.DataFrame(data))

        stim = pd.concat(slist, ignore_index=True)
        stim['go_trial'] = stim['tar_freq']!=stim['dis_freq']

        if self.switch_channels:
            d2=stim.copy()
            d2['tar_channel']=1-self.primary_channel
            stim = pd.concat([stim,d2], ignore_index=True)

        self.stim_list = stim.copy().reset_index()
        print(self.stim_list)
        total_wav_set = len(stim)

        # set up wav_set_idx to trial_idx mapping  -- self.trial_wav_idx
        if trial_idx is None:
            trial_idx = self.current_trial_idx

        if trial_idx >= len(self.trial_wav_idx):
            # hack to prevent identical sequences from repeating
            for t in range(trial_idx):
                _ = _rng.permutation(np.arange(total_wav_set, dtype=int))
            new_trial_wav = _rng.permutation(np.arange(total_wav_set, dtype=int))
            self.trial_wav_idx = np.concatenate((self.trial_wav_idx, new_trial_wav))
            log.info(f'Added {len(new_trial_wav)}/{len(self.trial_wav_idx)} trials to trial_wav_idx')
            self.current_full_rep += 1
            self.trial_is_repeat = np.concatenate((self.trial_is_repeat, np.zeros_like(new_trial_wav)))


    def trial_waveform(self, trial_idx=None, wav_set_idx=None):

        row = self.stim_row(trial_idx=trial_idx, wav_set_idx=wav_set_idx)

        wbins = int(row['duration']*self.fs)
        t=np.arange(wbins)/self.fs
        wfg = np.sin(t*2*np.pi*row['tar_freq'])*5
        env = np.abs(np.sin(t*2*np.pi*row['tar_am']/2))*2
        wfg *= env
        wbg = np.sin(t*2*np.pi*row['dis_freq'])*5

        fg_level = row['tar_level']
        bg_level = row['dis_level']
        if fg_level == 0:
            fg_scaleby = 0
        else:
            fg_scaleby = 10 ** ((fg_level - 80) / 20)
        if bg_level == 0:
            bg_scaleby = 0
        else:
            bg_scaleby = 10 ** ((bg_level - 80) / 20)
        wfg *= fg_scaleby
        wbg *= bg_scaleby

        # combine fg and bg waveforms
        if row['tar_channel'] == 0:
            w = np.stack((wfg, wbg), axis=1)
        else:
            w = np.stack((wbg, wfg), axis=1)
        print(row)
        log.info(f"fg level: {fg_level} bg level: {bg_level} FG RMS: {wfg.std():.3f} BG RMS: {wbg.std():.3f}")
        log.info(f"**** trial {trial_idx} wavidx {row['index']}  tar channel: {row['tar_channel']}")

        return w.T

    def trial_parameters(self, trial_idx=None, wav_set_idx=None):

        row = self.stim_row(trial_idx=trial_idx, wav_set_idx=wav_set_idx)

        is_go_trial = row['go_trial']
        if is_go_trial == 0:
            if (self.reward_ambiguous_frac == 0.5):
                # random
                response_condition = int(np.ceil(np.random.uniform(0, 2)))
            elif (self.reward_ambiguous_frac == 0):
                # none
                response_condition = 0
            else:
                # either
                response_condition = -1
        else:
            # 1=spout 1, 2=spout 2
            response_condition = int(row['tar_channel'] + 1)

        tar_name = f"{row['tar_freq']}:{row['tar_level']}:{row['tar_am']}"
        dis_name = f"{row['dis_freq']}:{row['dis_level']}"
        response_window = (self.response_window[0],self.response_window[1])
        log.info(f"**** trial {trial_idx} wavidx {row['index']} parms tar channel: {row['tar_channel']} "
                 f"response cond {response_condition}")

        d = {'trial_idx': trial_idx,
             'wav_set_idx': row['index'],
             'target_name': tar_name,
             'distractor_name': dis_name,
             'this_target_frequency': row['tar_freq'],
             'this_target_am': row['tar_am'],
             'this_distractor_frequency': row['dis_freq'],
             'this_duration': row['duration'],
             'this_target_level': row['tar_level'],
             'this_distractor_level': row['dis_level'],
             'this_snr': row['tar_level']-row['dis_level'],
             'response_condition': response_condition,
             'response_window': response_window,
             'current_full_rep': self.current_full_rep,
             'primary_channel': self.primary_channel,
             'trial_is_repeat': self.trial_is_repeat[trial_idx-1],
        }

        return d


class VowelSet(WavSet):

    default_parameters = [
        {'name': 'sound_path', 'label': 'folder', 'default': 'h:/sounds/vowels/v2',
         'dtype': 'str', 'scope': 'experiment'},
        {'name': 'target_set', 'label': 'Target names (list)',
         'expression': '["EH_106"]', 'dtype': 'object', 'scope': 'experiment'},
        {'name': 'non_target_set', 'label': 'Non-target names (list)',
         'expression': '["IH_106"]', 'dtype': 'object', 'scope': 'experiment'},
        {'name': 'catch_set', 'label': 'Catch names (list)', 'expression': '[]',
         'dtype': 'object', 'scope': 'experiment'},
        {'name': 'switch_channels', 'label': 'Targets from both sides?',
         'compact_label': 'combinations', 'default': 'No',
         'choices': {'No': "False", 'Yes': "True"},
         'scope': 'experiment', 'type': 'EnumParameter'},
        {'name': 'primary_channel', 'label': 'primary_channel',
         'compact_label': 'primary_channel', 'default': '0',
         'choices': {'0': 0, '1': 1},
         'scope': 'experiment', 'type': 'EnumParameter'},
        {'name': 'duration', 'label': 'duration of each sample (s)',
         'default': 0.24, 'dtype': 'double', 'scope': 'experiment'},
        {'name': 'repeat_count', 'label': 'repeats per trial', 'default': 2,
         'dtype': 'int', 'scope': 'experiment'},
        {'name': 'repeat_isi', 'label': 'ISI between repeats (s)',
         'default': 0.2, 'dtype': 'double', 'scope': 'experiment'},
        {'name': 'tar_to_cat_ratio', 'label': 'Ratio of tar to catch trials',
         'default': 5, 'dtype': 'int', 'scope': 'experiment'},
        {'name': 'level', 'label': 'level (dB peSPL)', 'default': 60,
         'dtype': 'double', 'scope': 'experiment'},
        {'name': 'fs', 'label': 'sampling rate (1/s)', 'default': 44000,
         'dtype': 'double', 'scope': 'experiment'},
        {'name': 'response_start', 'label': 'response win start (s)',
         'default': 0, 'dtype': 'double', 'scope': 'experiment'},
        {'name': 'response_end', 'label': 'response win end (s)', 'default': 2,
         'dtype': 'double', 'scope': 'experiment'},
        {'name': 'random_seed', 'label': 'random_seed', 'default': 0, 'dtype':
         'int', 'scope': 'experiment'},
        {'name': 's1_name', 'label': 'S1', 'type': 'Result', 'type': 'Result'},
        {'name': 's2_name', 'label': 'S2', 'type': 'Result'},
        {'name': 'stim_cat', 'label': 'Cat', 'type': 'Result'},
    ]

    for d in default_parameters:
        # Use `setdefault` so we don't accidentally override a parameter that
        # wants to use a different group.
        d.setdefault('group_name', 'VowelSet')

    def __init__(self, n_response, **parameter_dict):
        super().__init__(n_response=n_response)
        # internal object to handle wavs, don't need to specify independently
        log.info('N_response %r', self.n_response)
        self.duration = 0

        # trial management
        self.update_parameters(parameter_dict)

    def update_parameters(self, parameter_dict):
        for k, v in parameter_dict.items():
            setattr(self, k, v)
        self.response_window = (parameter_dict['response_start'], parameter_dict['response_end'])
        self.wavset = MCWavFileSet(
            fs=self.fs, path=self.sound_path, duration=self.duration,
            normalization='rms', fit_range=slice(0, None), test_range=None,
            test_reps=2, channel_count=1, level=self.level)

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

        row = self.stim_row(trial_idx=trial_idx, wav_set_idx=wav_set_idx)
        wav_set_idx = row['index']
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
        row = self.stim_row(trial_idx=trial_idx, wav_set_idx=wav_set_idx)
        wav_set_idx = row['index']

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
                if np.random.rand()>=0.5:
                    response_condition = 0
                else:
                    response_condition = 1

        response_window = self.response_window

        d = {'trial_idx': trial_idx,
             'wav_set_idx': wav_set_idx,
             's1idx': s1idx,
             's2idx': s2idx,
             's1_name': s1_name,
             's2_name': s2_name,
             'stim_cat': stim_cat,
             'duration': self.duration,
             'response_condition': response_condition,
             'response_window': response_window,
             'current_full_rep': self.current_full_rep,
             'primary_channel': self.primary_channel,
             'trial_is_repeat': self.trial_is_repeat[trial_idx-1],
             }
        return d

    def score_response_old(self, outcome, repeat_incorrect=True, trial_idx=None):
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
        stim_cat = self.stim_cat[wav_set_idx]

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


class OldCategorySet(FgBgSet):
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
        Notes on running different modes:
            > Use CatchFgSet=None, CatchBgSet=None, OAnoiseSet=None (or omit these params) to run simple
                ferret_voc vs speech (no noise, no catch/vocoding)
            > Use OAnoiseSet and unique_overall_SNR (and set CatchFgSet=None, CatchBgSet=None) to run
                ferret_voc vs speech in diotic noise
            > Use CatchBgSet and reg2catch_ratio (and set OAnoiseSet=None, CatchFgSet=None) to run
                ferret_voc vs speech in vocoding condition

        FgBgSet polls FgSet and BgSet for .max_index, .waveform, .names, and .fs
        :param FgSet: {WaveformSet, MultichannelWaveformSet, None}
        :param BgSet: {WaveformSet, MultichannelWaveformSet, None}
        :param CatchBgSet: {WaveformSet, MultichannelWaveformSet, None}
        :param OAnoiseSet: {WaveformSet, MultichannelWaveformSet, None}
            Represents the overall SNR of diotic noise. (Note: FGrms/BGrms = 1; i.e., 0 dB)
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
        self.primary_channel = primary_channel
        self._fg_delay = fg_delay
        self._fg_snr = fg_snr
        self.random_seed = random_seed

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

        self.current_trial_idx = 0

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

        assert len(self._fg_snr) == 1, "Only a single SNR is supported"
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
        log.info(f"wfg={wfg.shape},wbg={wbg.shape}, bfdur={self.BgSet.duration}, fg level: {self.FgSet.level} "
                 f"bg level: {self.BgSet.level} FG RMS: {wfg.std():.3f} BG RMS: {wbg.std():.3f}")

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

class OverlappingSounds(FgBgSet):

    def __init__(self, n_response, **parameter_dict):
        raise NotImplementedError('Placeholder')

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


##
## Passive wav_set stim
##

class BinauralTone(WavSet):
    """ Passive """
    default_parameters = [
        {'name': 'reference_center', 'label': 'Reference frequency',
         'expression': '1000', 'dtype': 'object', 'scope': 'experiment'},
        {'name': 'probe_octaves', 'label': 'Tone octaves (above/below ref)',
         'expression': '[1]', 'dtype': 'object', 'scope': 'experiment'},
        {'name': 'probe_count', 'label': 'Tone count (tiled over octaves)',
         'expression': '9', 'dtype': 'object', 'scope': 'experiment'},
        {'name': 'probe_level', 'label': 'Probe level(s) (list)',
         'expression': '[-20,-10,0,10,20]', 'dtype': 'object', 'scope': 'experiment'},
       {'name': 'reference_level', 'label': 'Reference dB SPL',
         'expression': '50', 'dtype': 'object', 'scope': 'experiment'},

        {'name': 'duration', 'label': 'duration of each sample (s)',
         'default': 0.1, 'dtype': 'double', 'scope': 'experiment'},
        {'name': 'pre_silence', 'label': 'pre-stim silence (s)',
         'default': 0.05, 'dtype': 'double', 'scope': 'experiment'},
        {'name': 'post_silence', 'label': 'post-stim silence (s)',
         'default': 0.05, 'dtype': 'double', 'scope': 'experiment'},

        {'name': 'primary_channel', 'label': 'Primary (contra) channel',
         'compact_label': 'primary_channel', 'default': '0',
         'choices': {'0': 0, '1': 1},
         'scope': 'experiment', 'type': 'EnumParameter'},
        {'name': 'switch_channels', 'label': 'Switch ref channel?',
         'compact_label': 'combinations', 'default': 'No',
         'choices': {'No': "False", 'Yes': "True"},
         'scope': 'experiment', 'type': 'EnumParameter'},

        {'name': 'fs', 'label': 'sampling rate (1/s)', 'default': 44000,
         'dtype': 'double', 'scope': 'experiment'},
        {'name': 'ramp', 'label': 'on/off ramp (ms)', 'default': 5,
         'dtype': 'double', 'scope': 'experiment'},
        {'name': 'random_seed', 'label': 'random_seed', 'default': 0, 'dtype':
         'int', 'scope': 'experiment'},
        {'name': 'this_reference_frequency', 'label': 'R', 'type': 'Result'},
        {'name': 'this_probe_frequency', 'label': 'P', 'type': 'Result'},
        {'name': 'this_snr', 'label': 'level', 'type': 'Result'},
        {'name': 'current_full_rep', 'label': 'rep', 'type': 'Result'},
    ]

    for d in default_parameters:
        # Use `setdefault` so we don't accidentally override a parameter that
        # wants to use a different group.
        d.setdefault('group_name', 'AMFusion')

    def __init__(self, n_response=0, **parameter_dict):
        super().__init__(n_response=n_response)

        self.update_parameters(parameter_dict)

    def update(self, trial_idx=None):
        """figure out indexing to map wav_set idx to specific members of FgSet and BgSet.
        manage trials separately to allow for repeats, etc."""
        _rng = np.random.RandomState(self.random_seed)

        logref = np.log2(self.reference_center)
        loglo = logref - self.probe_octaves
        loghi = logref + self.probe_octaves

        frequency_range = np.round(2**np.linspace(loglo, loghi, self.probe_count))

        x, y, z = np.meshgrid(frequency_range, frequency_range, self.probe_level)
        ref = x.flatten()
        probe = y.flatten()
        level = z.flatten()
        names = [f"{r}:{p}:{l}" for r,p,l in zip(ref,probe,level)]
        data = {
            'name': names,
            'reference_frequency': ref,
            'probe_frequency': probe,
            'probe_level': level,
            'duration': self.duration,
            'tar_channel': self.primary_channel,
        }
        stim = pd.DataFrame(data)

        if self.switch_channels:
            d2 = stim.copy()
            d2['tar_channel'] = 1 - self.primary_channel
            stim = pd.concat([stim, d2], ignore_index=True)

        stim['index'] = stim.index
        self.stim_list = stim.copy()

        total_wav_set = len(stim)

        # set up wav_set_idx to trial_idx mapping  -- self.trial_wav_idx
        if trial_idx is None:
            trial_idx = self.current_trial_idx

        if trial_idx >= len(self.trial_wav_idx):
            # hack to prevent identical sequences from repeating
            for t in range(trial_idx):
                _ = _rng.permutation(np.arange(total_wav_set, dtype=int))
            new_trial_wav = _rng.permutation(np.arange(total_wav_set, dtype=int))
            self.trial_wav_idx = np.concatenate((self.trial_wav_idx, new_trial_wav))
            log.info(f'Added {len(new_trial_wav)}/{len(self.trial_wav_idx)} trials to trial_wav_idx')
            self.current_full_rep += 1
            self.trial_is_repeat = np.concatenate((self.trial_is_repeat, np.zeros_like(new_trial_wav)))

    @lru_cache(maxsize=None)
    def generate_tone(self, duration, frequency, level):
        wbins = int(duration*self.fs)
        t=np.arange(wbins)/self.fs
        wfg = np.sin(t*2*np.pi*frequency)*5

        rampbins = int(self.ramp * self.fs / 1000)
        onramp = np.linspace(0,1,rampbins)
        offramp = np.linspace(1,0,rampbins)
        wfg[:rampbins] = wfg[:rampbins] * onramp
        wfg[-rampbins:] = wfg[-rampbins:] * offramp

        if level == 0:
            fg_scaleby = 0
        else:
            fg_scaleby = 10 ** ((level - 80) / 20)

        wfg *= fg_scaleby

        return wfg


    def trial_waveform(self, trial_idx=None, wav_set_idx=None):

        row = self.stim_row(trial_idx=trial_idx, wav_set_idx=wav_set_idx)

        # wbins = int(row['duration']*self.fs)
        # t=np.arange(wbins)/self.fs
        # wfg = np.sin(t*2*np.pi*row['reference_frequency'])*5
        # wbg = np.sin(t*2*np.pi*row['probe_frequency'])*5
        #
        # rampbins = int(self.ramp * self.fs / 1000)
        # onramp = np.linspace(0,1,rampbins)
        # offramp = np.linspace(1,0,rampbins)
        # wfg[:rampbins] = wfg[:rampbins] * onramp
        # wfg[-rampbins:] = wfg[-rampbins:] * offramp
        # wbg[:rampbins] = wbg[:rampbins] * onramp
        # wbg[-rampbins:] = wbg[-rampbins:] * offramp
        #
        fg_level = self.reference_level
        bg_level = self.reference_level + row['probe_level']
        wfg = self.generate_tone(row['duration'], row['reference_frequency'], fg_level)
        wbg = self.generate_tone(row['duration'], row['probe_frequency'], bg_level)

        # combine fg and bg waveforms
        if row['tar_channel'] == 0:
            w = np.stack((wfg, wbg), axis=1)
        else:
            w = np.stack((wbg, wfg), axis=1)

        prebins, postbins = int(self.fs*self.pre_silence), int(self.fs*self.post_silence)
        wpre, wpost = np.zeros((prebins, 2)), np.zeros((postbins, 2))
        w = np.concatenate([wpre,w,wpost], axis=0)

        log.info(f"fg level: {fg_level} bg level: {bg_level} FG RMS: {wfg.std():.3f} BG RMS: {wbg.std():.3f}")
        log.info(f"**** trial {trial_idx} wavidx {row['index']}  tar channel: {row['tar_channel']}")

        return w.T

    def trial_parameters(self, trial_idx=None, wav_set_idx=None):

        row = self.stim_row(trial_idx=trial_idx, wav_set_idx=wav_set_idx)

        d = {'trial_idx': trial_idx,
             'wav_set_idx': row['index'],
             'this_name': row['name'],
             'this_reference_frequency': row['reference_frequency'],
             'this_probe_frequency': row['probe_frequency'],
             'this_snr': row['probe_level'],
             'current_full_rep': self.current_full_rep,
             'reference_channel': row['tar_channel'],
             'trial_is_repeat': self.trial_is_repeat[trial_idx-1],
        }

        return d


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

    default_parameters = [
        # Stim/paradigm params
        {'name': 'fg_path', 'label': 'FG path',
         'default': 'h:/sounds/Categories/v3_vocoding/', 'dtype': 'str'},
        {'name': 'fg_range', 'label': 'FG wav indexes', 'expression': '[-1]'},

        {'name': 'bg_path', 'label': 'BG path',
         'default': 'h:/sounds/Categories/speech_stims/', 'dtype': 'str'},
        {'name': 'bg_range', 'label': 'BG wav indexes', 'expression': '[-1]'},

        {'name': 'catch_fg_path', 'label': 'CatchFG path',
         'default': '', 'dtype': 'str'},
        {'name': 'catch_fg_range', 'label': 'CatchFG wav indexes', 'expression': '[-1]'},

        {'name': 'catch_bg_path', 'label': 'CatchBG path',
         'default': 'h:/sounds/Categories/chimeric_voc/', 'dtype': 'str'},
        {'name': 'catch_bg_range', 'label': 'CatchBG wav indexes', 'expression': '[-1]'},

        {'name': 'catch_ferret_id', 'label': 'Catch ferret', 'expression': '[4]'},
        {'name': 'n_env_bands', 'label': 'N-band(s) vocoding', 'expression': '[2, 8, 32]'},
        {'name': 'reg2catch_ratio', 'label': 'Ratio (Reg/Catch)', 'default': 6, 'dtype': 'int'},

        {'name': 'OAnoise_path', 'label': 'OAnoise path',
         'default': 'h:/sounds/Categories/noise_vocPSDmatched/', 'dtype': 'str'},
        {'name': 'OAnoise_range', 'label': 'OAnoise wav indexes', 'expression': '[-1]'},

        {'name': 'normalization', 'label': 'Normalization', 'default': 'rms', 'type': 'EnumParameter',
         'choices': {'max': "'pe'", 'RMS': "'rms'", 'fixed': "'fixed'"}},
        {'name': 'duration', 'label': 'FG/BG duration (s)', 'default': 3.0, 'dtype': 'float'},
        {'name': 'norm_fixed_scale', 'label': 'fixed norm value', 'default': 1, 'dtype': 'float'},
        {'name': 'fg_level', 'label': 'FG level(s) dBSPL', 'expression': '[55]', 'dtype': 'object'},
        {'name': 'bg_level', 'label': 'BG level(s) dBSPL', 'expression': '[55]', 'dtype': 'object'},
        {'name': 'OAnoise_SNR', 'label': 'OA noise SNR(s) dB', 'expression': '[np.inf]', 'dtype': 'object'},

        {'name': 'fg_delay', 'label': 'FG delay (s)', 'default': 0.0, 'dtype': 'float'},
        {'name': 'primary_channel', 'label': 'Primary FG channel', 'default': 0, 'dtype': 'int'},
        {'name': 'fg_switch_channels', 'label': 'Switch FG channel', 'type': 'BoolParameter', 'default': True},
        {'name': 'combinations', 'label': 'How to combine FG+BG', 'default': 'custom',
         'type': 'EnumParameter', 'choices': {'simple': "'simple'", 'all': "'all'", 'custom': "'custom'"}},

        # {'name': 'contra_n', 'label': 'Contra BG portion (int)', 'default': 1, 'dtype': 'int'},
        # {'name': 'diotic_n', 'label': 'Diotic BG portion (int)', 'default': 0, 'dtype': 'int'},
        # {'name': 'ipsi_n', 'label': 'Ipsi BG portion (int)', 'default': 0, 'dtype': 'int'},

        {'name': 'migrate_fraction', 'label': 'Percent migrate trials', 'default': '0',
         'type': 'EnumParameter', 'choices': {'0': 0.0, '25': 0.25, '50': 0.5}},
        {'name': 'migrate_start', 'label': "migrate_start (s)", 'default': 0.5, 'dtype': 'float'},
        {'name': 'migrate_stop', 'label': "migrate_stop (s)", 'default': 1.0, 'dtype': 'float'},

        {'name': 'response_window', 'label': 'Response start,stop (s)', 'expression': '(0, 1)'},
        {'name': 'reward_durations', 'label': 'FG reward durations', 'expression': '()'},

        {'name': 'random_seed', 'label': 'Random seed', 'default': 0, 'dtype': 'int'},

        # Unsure about the following
        {'name': 'fg_choice_trials', 'label': 'FG choice portion (int)', 'default': 0, 'dtype': 'int'},
        {'name': 'reward_ambiguous_frac', 'label': 'Frac. reward ambiguous', 'default': 'all',
         'type': 'EnumParameter', 'choices': {'all': 1.0, 'random 50%': 0.5, 'never': 0.0}},
        {'name': 'fs', 'label': 'Sampling rate (Hz)', 'default': 44000},

        # Results
        # {'name': 'fs', 'label': 'Sampling rate (Hz)', 'default': 44000, 'group_name': 'Results'},

        {'name': 'fg_channel', 'label': 'FG chan', 'type': 'Result', 'group_name': 'Results'},
        {'name': 'bg_channel', 'label': 'BG chan', 'type': 'Result', 'group_name': 'Results'},
        {'name': 'fg_name', 'label': 'FG', 'type': 'Result', 'group_name': 'Results'},
        {'name': 'bg_name', 'label': 'BG', 'type': 'Result', 'group_name': 'Results'},
        {'name': 'this_snr', 'label': 'Trial SNR', 'type': 'Result', 'group_name': 'Results'},
        {'name': 'migrate_trial', 'label': 'Moving Tar', 'type': 'Result', 'group_name': 'Results'},
        {'name': 'current_full_rep', 'label': 'Rep', 'type': 'Result', 'group_name': 'Results'},
        {'name': 'trial_cat', 'label': 'Type', 'type': 'Result', 'group_name': 'Results'},

    ]
    for d in default_parameters:
        # Use `setdefault` so we don't accidentally override a parameter that
        # wants to use a different group.
        d.setdefault('group_name', 'CategorySet')


    def __init__(self, **parameter_dict):
        """
        Notes on running different modes:
            > Use CatchFgSet=None, CatchBgSet=None, OAnoiseSet=None (or omit these params) to run simple
                ferret_voc vs speech (no noise, no catch/vocoding)
            > Use OAnoiseSet and OAnoise_SNR (and set CatchFgSet=None, CatchBgSet=None) to run
                ferret_voc vs speech in diotic noise
            > Use CatchBgSet and reg2catch_ratio (and set OAnoiseSet=None, CatchFgSet=None) to run
                ferret_voc vs speech in vocoding condition
        """

        # super().__init__(n_response=n_response)
        # internal object to handle wavs, don't need to specify independently
        self.n_response = 2 # because two spouts (left vs right)
        log.info('N_response %r', self.n_response)
        self.duration = 0

        # left out params from OldCategorySet (maybe don't need these)
        # bg_switch_channels = False, fg_snr = 0.0

        self.current_trial_idx = 0

        # trial management
        self.trial_wav_idx = np.array([], dtype=int)
        self.trial_outcomes = np.array([], dtype=int)
        self.trial_is_repeat = np.array([], dtype=int)
        self.current_full_rep = 0
        self.stim_list = pd.DataFrame()

        # trial management
        self.update_parameters(parameter_dict)


    def update_parameters(self, parameter_dict):
        for k, v in parameter_dict.items():
            setattr(self, k, v)

        FgSet = MCWavFileSet(fs=self.fs, path=self.fg_path, duration=self.duration, channel_count=1,
                                  fit_range=self.fg_range, normalization=self.normalization, level=self.fg_level)
        BgSet = MCWavFileSet(fs=self.fs, path=self.bg_path, duration=self.duration, channel_count=1,
                                  fit_range=self.bg_range, normalization=self.normalization, level=self.bg_level)
        if self.catch_fg_path!='':
            CatchFgSet = MCWavFileSet(
                fs=self.fs, path=self.catch_fg_path, duration=self.duration, level=self.fg_level,
                normalization=self.normalization, fit_range=self.catch_fg_range, channel_count=1)
        else:
            CatchFgSet = None

        if self.catch_bg_path != '':
            CatchBgSet = MCWavFileSet(
                fs=self.fs, path=self.catch_bg_path, duration=self.duration, level=self.bg_level,
                normalization=self.normalization, fit_range=self.catch_bg_range, channel_count=1)
        else:
            CatchBgSet = None

        assert self.fg_level==self.bg_level, ("OAnoise_level calculation is set to fg_level+OAnoise_SNR "
                                              "assuming fg_level==bg_level. If you must use different FG/BG "
                                              "levels, then use noise level = OAnoise_SNR + "
                                              "pow2db(db2pow(fg_level)+db2pow(bg_level))")

        if self.OAnoise_path != '' and not all(np.isinf(x) for x in self.OAnoise_SNR):
            noise_level = list(self.fg_level - np.array(self.OAnoise_SNR))
            OAnoiseSet = MCWavFileSet(
                fs=self.fs, path=self.OAnoise_path, duration=self.duration, level=noise_level,
                normalization=self.normalization, fit_range=self.OAnoise_range, channel_count=1)
        else:
            OAnoiseSet = None


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
            if not all(np.isinf(self.OAnoise_SNR)):
                raise ValueError(f"OAnoise_SNR must be [np.inf] if OAnoiseSet is None")
        else:
            self.OAnoiseSet = OAnoiseSet

        self.update()

    @property
    def wav_per_rep(self):
        return len(self.stim_list)

    # inherit from FgBgSet
    def update(self, trial_idx=None):
        """figure out indexing to map wav_set idx to specific members of FgSet and BgSet.
        manage trials separately to allow for repeats, etc."""
        _rng = np.random.RandomState(self.random_seed)
        np.random.seed(self.random_seed)

        if all(np.isinf(self.OAnoise_SNR)) and (self.catch_bg_path != ''):
            # There are catch trials
            fgi, bgi, fgg = get_stim_list(self.FgSet, self.BgSet, self.catch_ferret_id,
                                          self.n_env_bands, self.reg2catch_ratio)
            overall_snr = np.inf * np.ones_like(fgi)

        else:
            fgi, bgi, fgg = get_stim_list_no_catch(self.FgSet, self.BgSet)
            # add np.inf as the first snr if doesn't already exist
            if all(x == np.inf for x in self.OAnoise_SNR):
                overall_snr = np.inf * np.ones_like(fgi)
            else:
                self.OAnoise_SNR = [np.inf] + list(set(self.OAnoise_SNR) - set([np.inf]))
                num_noninf_snrs = len(self.OAnoise_SNR) - 1
                overall_snr_weight = np.concatenate(([num_noninf_snrs], np.ones(num_noninf_snrs)))
                overall_snr_prob = overall_snr_weight / overall_snr_weight.sum()
                # overall_snr = choices(self.OAnoise_SNR, weights = overall_snr_weight, k = len(fgi))
                overall_snr = np.random.choice(self.OAnoise_SNR, p=overall_snr_prob, size=len(fgi))

        num_uniq_trial = len(fgi)

        # region set fg an bg
        if self.fg_switch_channels:
            fgc = np.concatenate((np.zeros_like(fgi), np.ones_like(fgi)))
            fgi, bgi, fgg, overall_snr = (np.tile(fgi, 2), np.tile(bgi, 2),
                                          np.tile(fgg, 2), np.tile(overall_snr, 2))
            num_uniq_trial *= 2
        else:
            fgc = np.zeros_like(fgi) + self.primary_channel

        bgc = 1 - fgc

        # if self.bg_switch_channels == False:
        #     bgc = np.zeros_like(fgi)
        # elif self.bg_switch_channels == 'same':
        #     bgc = fgc.copy()
        # elif self.bg_switch_channels == 'opposite':
        #     bgc = 1 - fgc
        # else:
        #     raise ValueError(f"Unknown bg_switch_channels value: {self.bg_switch_channels}.")
        # print(f"{self._fg_snr}: type = {type(self._fg_snr)}")
        # assert len(self._fg_snr) == 1, "Only a single SNR is supported"
        fg2bg_snr = self.fg_level-self.bg_level
        fsnr = np.zeros_like(fgi) + fg2bg_snr
        migrate_trial = np.zeros_like(fgi)
        total_wav_set = len(fgg)

        if (type(self.fg_delay) is np.array) | (type(self.fg_delay) is list):
            self.fg_delay = np.array(self.fg_delay)
        else:
            self.fg_delay = np.zeros_like(fgi) + self.fg_delay

        # Important
        # self.bg_index = bgi
        # self.fg_index = fgi
        # self.fg_channel = fgc
        # self.bg_channel = bgc
        # self.fg_snr = fsnr
        # self.fg_go = fgg
        # self.migrate_trial = migrate_trial
        # self.overall_snr = overall_snr

        data = {
            'bg_index': bgi,
            'fg_index': fgi,
            'fg_channel': fgc,
            'bg_channel': bgc,
            'fg_snr': fsnr,
            'fg_go': fgg,
            'migrate_trial': migrate_trial,
            'overall_snr': overall_snr,
            'fg_delay': self.fg_delay,
        }
        self.stim_list = pd.DataFrame(data)

        # ------- Important


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
        row = self.stim_row(trial_idx=trial_idx, wav_set_idx=wav_set_idx)
        wfg = self.FgSet.waveform(row['fg_index'])
        if row['fg_channel'] == 1:
            wfg = np.concatenate((np.zeros_like(wfg), wfg), axis=1)
        wbg = self.BgSet.waveform(row['bg_index'])
        log.info(f"wfg={wfg.shape},wbg={wbg.shape}, bfdur={self.BgSet.duration},fg level: {self.FgSet.level} "
                 f"bg level: {self.BgSet.level} FG RMS: {wfg.std():.3f} BG RMS: {wbg.std():.3f}")

        if row['bg_channel'] == 1:
            wbg = np.concatenate((np.zeros_like(wbg), wbg), axis=1)
        elif row['bg_channel'] == -1:
            wbg = np.concatenate((wbg, wbg), axis=1)

        if wbg.shape[1] < wfg.shape[1]:
            wbg = np.concatenate((wbg, np.zeros_like(wbg)), axis=1)
        if wfg.shape[1] < wbg.shape[1]:
            wfg = np.concatenate((wfg, np.zeros_like(wfg)), axis=1)
        fg_snr = row['fg_snr']
        if fg_snr == -100:
            fg_scale = 0
        elif fg_snr < 50:
            fg_scale = 10**(fg_snr / 20)
        else:
            # special case of effectively infinite SNR, don't actually amplify fg
            wbg[:] = 0
            fg_scale = 10**((fg_snr-100) / 20)

        offsetbins = int(self.fg_delay[row['fg_index']] * self.FgSet.fs)
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

        if not np.isinf(row['overall_snr']):
            cur_overall_snr = row['overall_snr']
            overall_noise_scale = 10 ** (-cur_overall_snr / 20)
            # noise and ferret vocal can be matched in index
            ov_noise = self.OAnoiseSet.waveform(row['fg_index'])
            ov_noise = np.concatenate((ov_noise, ov_noise), axis=1)
            sig_len = w.shape[0]
            ov_noise = ov_noise[:sig_len, :]
            log.info(f"<flag><><><><><><>< {trial_idx}|{wav_set_idx}| "
                     f"snr = {cur_overall_snr}: w={w.shape}, ov_noise{ov_noise.shape}")
            w = w/overall_noise_scale + ov_noise  # will be ~3 dB louder than clean

        return w.T

    def trial_parameters(self, trial_idx=None, wav_set_idx=None):
        row = self.stim_row(trial_idx=trial_idx, wav_set_idx=wav_set_idx)
        fg_i = row['fg_index']
        bg_i = row['bg_index']

        is_go_trial = row['fg_go']
        if is_go_trial == -1:
            # -1 means either port
            response_condition = -1
        elif is_go_trial == 1:
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
             'snr': row['fg_snr'],
             'this_snr': row['fg_snr'],
             'overall_snr': row['overall_snr'],
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

        #def score_response(self, outcome, repeat_incorrect=2, trial_idx=None):
