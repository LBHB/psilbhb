from functools import partial, lru_cache
import itertools
from pathlib import Path

import numpy as np
from scipy import signal
from scipy.io import wavfile
import pandas as pd

from copy import deepcopy
import os

from psiaudio import util
from .wav_set import WavSet, MCWavFileSet, FgBgSet
from .basic_sounds import generate_tone

import logging
log = logging.getLogger(__name__)

# TODO SOMEDAY SPLIT CONTENTS OF wav_set.py out so that everything is not in one ginormous file