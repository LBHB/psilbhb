import logging
import contextlib

from functools import partial, lru_cache

import numpy as np
from scipy import signal
from scipy.io import wavfile

from joblib import Memory


log = logging.getLogger(__name__)

location = 'cachedir'
memory = Memory(location, verbose=0)

@contextlib.contextmanager
def temp_seed(seed):
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


@memory.cache
def generate_tone(duration, frequency, level=60, fs=44000, ramp=5, calibration=None):
    """
    Generate tone
    :param duration:
    :param frequency:
    :param level: level in dB SPL (80 dB = +/-5), default 60
    :param fs: (Hz) default 44000
    :param ramp: (ms) on/off ramp, default 5 ms
    :return: w: np.array of length int(duration*fs)
    """
    wbins = int(duration*fs)
    t = np.arange(wbins)/fs
    w = np.sin(t*2*np.pi*frequency)

    # apply ramp
    rampbins = int(ramp * fs / 1000)
    onramp = np.linspace(0, 1, rampbins)
    offramp = np.linspace(1, 0, rampbins)
    w[:rampbins] = w[:rampbins] * onramp
    w[-rampbins:] = w[-rampbins:] * offramp

    if calibration is not None:
        scaleby = calibration.get_sf(frequency, level) * np.sqrt(2)
        log.error('TONE %f %f fg_scaleby %f', frequency, level, scaleby / np.sqrt(2))
    elif level == 0:
        scaleby = 0
    else:
        scaleby = 10 ** ((level - 80) / 20) * 5

    w *= scaleby

    return w

def generate_am_tone(duration, frequency, level=60, fs=44000, ramp=5, calibration=None):
    """
    Generate tone
    :param duration:
    :param frequency:
    :param level: level in dB SPL (80 dB = +/-5), default 60
    :param fs: (Hz) default 44000
    :param ramp: (ms) on/off ramp, default 5 ms
    :return: w: np.array of length int(duration*fs)
    """
    hcount = len(harmonics)
    ref_level = row['ref_level']
    prb_level = row['ref_level'] + row['prb_level']
    wfg = generate_tone(row['duration'], row['ref_frequency'], ref_level,
                        fs=self.fs, ramp=self.ramp, calibration=calibration)

    wbins = int(self.duration * self.fs)
    bgduration = row['duration'] - row['prb_delay'] / 1000
    bgbins = int(bgduration * self.fs)

    wfg = np.zeros(wbins)
    wbg = np.zeros(bgbins)
    for h in harmonics:
        if ref_level - prb_level > -60:
            wfg += generate_tone(row['duration'], row['ref_frequency'] * (h + 1), ref_level, fs=self.fs,
                                 ramp=self.ramp) / hcount
        if ref_level - prb_level < 60:
            wbg += generate_tone(bgduration, row['prb_frequency'] * (h + 1), prb_level, fs=self.fs,
                                 ramp=self.ramp) / hcount
    padbins = len(wfg) - len(wbg)
    if padbins > 0:
        wbg = np.concatenate((np.zeros(padbins, dtype=wbg.dtype), wbg))

    if row['ref_am'] > 0:
        t = np.arange(wbins) / self.fs
        env = 1 + np.sin(t * 2 * np.pi * row['ref_am']) * 10 ** (-row['ref_moddepth'] / 20)
        wfg *= env


    wbins = int(duration*fs)
    t = np.arange(wbins)/fs
    w = np.sin(t*2*np.pi*frequency)*5

    rampbins = int(ramp * fs / 1000)
    onramp = np.linspace(0, 1, rampbins)
    offramp = np.linspace(1, 0, rampbins)
    w[:rampbins] = w[:rampbins] * onramp
    w[-rampbins:] = w[-rampbins:] * offramp

    if level == 0:
        fg_scaleby = 0
    else:
        fg_scaleby = 10 ** ((level - 80) / 20)

    w *= fg_scaleby

    return w


@memory.cache
def generate_tone_stack(freq, f_offsets, phases, target_bandwidth, db_depth, am, duration, fs):

    hcount = len(f_offsets)

    # generate the carriers
    wbins = int(duration * fs)
    t = np.arange(wbins) / fs
    w = np.zeros(wbins)
    for h, ph in zip(f_offsets, phases):
        # log.info(f"{h:.3f} {row['tar_freq'] * (h+1)}")
        w += np.sin(t * 2 * np.pi * freq * (h + 1) + ph) * (5 / hcount)
    if target_bandwidth > 0:
        # fix RMS level to be 80 dB
        w = w / w.std() * 3.5349

    depth = -np.abs(10**(db_depth/20))
    if am>0:
        env = (1 + np.sin(t*2*np.pi*am) * depth)
        w = w * env / 2

    return w