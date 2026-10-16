import logging
log = logging.getLogger(__name__)

import numpy as np
from psilbhb.stim.wav_set import load_wav


class ContinuousWavSet:

    default_parameters = []

    def __init__(self, n_response, output_cal, **parameter_dict):
        self.output_cal = output_cal
        self.n_output = len(output_cal)
        self.parameters = parameter_dict


class SilenceContinuousWavSet(ContinuousWavSet):

    default_parameters = [
        {'name': 'equalize', 'label': 'Apply equalizer?',
         'choices': {'No': "False", 'Yes': "True"}, 'default': 'No',
         'scope': 'experiment', 'type': 'EnumParameter', 'group_name': 'Continuous Silence'},
    ]

    def next(self, samples, channel):
        return np.zeros(samples)


class NaturalSequenceWavSet(ContinuousWavset):

    default_parameters = [
        {
            'name': 'equalize', 
            'label': 'Apply equalizer?',
            'choices': {'No': "False", 'Yes': "True"}, 
            'default': 'No',
            'scope': 'experiment', 
            'type': 'EnumParameter', 
            'group_name': 'Natural Sequence',
        },
        {
            'name': 'path', 
            'label': 'Path to sounds',
            'scope': 'experiment', 
            'group_name': 'Natural Sequence',
        },
        {
            'name': 'level', 
            'label': 'Stimulus level (dB SPL)',
            'scope': 'experiment', 
            'group_name': 'Natural Sequence',
        },
        {
            'name': 'fs', 
            'label': 'Sampling rate (sec^-1)', 
            'default': 44000, 
            'group_name': 'Natural Sequence'
        },
    ]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        wav_filenames = sorted(Path(self.parameters['path']).glob('*.wav'))
        self.waveforms = []
        for filename in wav_filenames:
            w = load_wav(
                    self.parameters['fs'], 
                    filename, 
                    self.parameters['level'], 
            )
            self.waveforms.append(w)

        for output_cal in 
        self.queue = queue.BlockedRandomSignalQueue(fs=self.fs, seed=self.random_seed)
        #self.queue = queue.BlockedRandomSignalQueue(self.fs, self.random_seed)
        metadata = [{'filename': w.filename.stem} for w in self.wav_files]
        self.queue.extend(self.wav_files, np.inf, duration=self.duration, metadata=metadata)

        queues = {}

        for i in range(self.N_output):
            queues[i] = queue.BlockedRandomSignalQueue(fs=self.fs, seed=self.parameters'random_seed'])
            self.queue = queue.Blocke
