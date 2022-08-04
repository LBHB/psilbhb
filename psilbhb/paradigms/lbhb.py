from psi.experiment.api import ParadigmDescription


PATH = 'psilbhb.paradigms.behavior.'
CORE_PATH = 'psi.paradigms.core.'


COMMON_PLUGINS = [
    {'manifest': PATH + 'behavior_gonogo.BehaviorManifest'},
    {'manifest': PATH + 'behavior_mixins.BaseGoNogoMixin'},
    {'manifest': PATH + 'behavior_mixins.WaterBolusDispenser'},
    {'manifest': 'psilbhb.paradigms.video.PSIVideo'},
    {'manifest': CORE_PATH + 'signal_mixins.SignalFFTViewManifest',
        'attrs': {'fft_time_span': 1, 'fft_freq_lb': 5, 'fft_freq_ub': 24000, 'y_label': 'Level (dB)'}
        },
]


ParadigmDescription(
    'NTD', 'Go-nogo tone detection in natural background',
    'animal', COMMON_PLUGINS + [
        {'manifest': PATH + 'stimuli.ToneInNaturalSounds'},
    ],
)


ParadigmDescription(
    'STD', 'Go-nogo tone detection in silence',
    'animal', COMMON_PLUGINS + [
        {'manifest': PATH + 'stimuli.ToneInSilence'},
    ],
)
