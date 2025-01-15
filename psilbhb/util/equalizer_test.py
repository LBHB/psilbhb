from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal

from cftsdata.iec import IEC

from psiaudio.calibration import InterpCalibration
# Needed to load octave scale
import psiaudio.plot
from psiaudio.stim import apply_max_correction
from psiaudio import util

experiment_folder = Path('D:\cfts\20241120-134543 Jereme Stinkysquid right pilot inear_speaker_calibration_chirp')
#experiment_folder = Path('/volume1/data/physiology/pilot/20241120-IEC-ferret/20241120-134726 Jereme Stinkysquid right pilot inear_speaker_calibration_chirp')
sens_file = experiment_folder / 'chirp_sens.csv'
sens = pd.read_csv(sens_file, index_col=['hw_ao_chirp_level', 'frequency'])

fh = IEC(experiment_folder)

# Load the microphone recordings to each individual presentation of the chirp
epochs = fh.get_epochs()

# Average across all presentations for each chirp level
epochs_mean = epochs.groupby('hw_ao_chirp_level').mean()

# Convert to PSD
epochs_mean_psd = util.psd_df(epochs_mean, fs=fh.hw_ai.fs)

# Load the calibration of the microphone that's embedded in the starship
mic_cal = fh.hw_ai.get_calibration()

# Convert PSD to SPL
epochs_mean_spl = mic_cal.get_db(epochs_mean_psd)