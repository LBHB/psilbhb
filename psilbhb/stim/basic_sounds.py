from functools import partial, lru_cache

import numpy as np
from scipy import signal
from scipy.io import wavfile


@lru_cache(maxsize=None)
def generate_tone(duration, frequency, level=60, fs=44000, ramp=5):
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
