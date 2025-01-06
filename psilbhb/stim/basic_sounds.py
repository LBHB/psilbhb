import logging
log = logging.getLogger(__name__)

from functools import partial, lru_cache

import numpy as np
from scipy import signal
from scipy.io import wavfile


@lru_cache(maxsize=None)
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

    rampbins = int(ramp * fs / 1000)
    onramp = np.linspace(0, 1, rampbins)
    offramp = np.linspace(1, 0, rampbins)
    w[:rampbins] = w[:rampbins] * onramp
    w[-rampbins:] = w[-rampbins:] * offramp
    calibration = None

    if calibration is not None:
        fg_scaleby = calibration.get_sf(frequency, level) * np.sqrt(2)
        log.error('TONE %f %f fg_scaleby %f', frequency, level, fg_scaleby / np.sqrt(2))
    if level == 0:
        fg_scaleby = 0
    else:
        fg_scaleby = 10 ** ((level - 80) / 20) * 5

    w *= fg_scaleby

    return w

def generate_am_tone(duration, frequency, level=60, fs=44000, ramp=5):
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
    fg_level = row['ref_level']
    bg_level = row['ref_level'] + row['prb_level']
    wfg = generate_tone(row['duration'], row['ref_frequency'], fg_level, fs=self.fs, ramp=self.ramp)

    wbins = int(self.duration * self.fs)
    bgduration = row['duration'] - row['prb_delay'] / 1000
    bgbins = int(bgduration * self.fs)

    wfg = np.zeros(wbins)
    wbg = np.zeros(bgbins)
    for h in harmonics:
        if fg_level - bg_level > -60:
            wfg += generate_tone(row['duration'], row['ref_frequency'] * (h + 1), fg_level, fs=self.fs,
                                 ramp=self.ramp) / hcount
        if fg_level - bg_level < 60:
            wbg += generate_tone(bgduration, row['prb_frequency'] * (h + 1), bg_level, fs=self.fs,
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
