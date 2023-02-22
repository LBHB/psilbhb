from psi.experiment.api import ParadigmDescription


PATH = 'psilbhb.paradigms.behavior.'
CORE_PATH = 'psi.paradigms.core.'


COMMON_PLUGINS = [
    {'manifest': PATH + 'behavior_mixins.BaseGoNogoMixin'},
    {'manifest': PATH + 'behavior_mixins.WaterBolusDispenser'},
    {'manifest': 'psilbhb.paradigms.video.PSIVideo'},
    {'manifest': 'psilbhb.paradigms.openephys.OpenEphysManifest'},
        {'manifest': PATH + 'behavior_mixins.SignalFFTViewManifest',
        'attrs': {'fft_time_span': 1, 'fft_freq_lb': 5, 'fft_freq_ub': 24000,
                'y_label': 'Level (dB)'}
            },
]


ParadigmDescription(
    'NTD-gonogo', 'Go-nogo tone detection in natural background',
    'animal', COMMON_PLUGINS + [
        {'manifest': PATH + 'behavior_gonogo.AutoBehaviorManifest'},
        {'manifest': PATH + 'stimuli.ToneInNaturalSoundsGoNogo'},
    ],
)


ParadigmDescription(
    'NTD-gonogo-np', 'Go-nogo tone detection in natural background (initiated)',
    'animal', COMMON_PLUGINS + [
        {'manifest': PATH + 'behavior_gonogo.InitiatedBehaviorManifest'},
        {'manifest': PATH + 'stimuli.ToneInNaturalSoundsGoNogo'},
    ],
)


ParadigmDescription(
    'NTD-2AFC', 'Two AFC tone detection in natural background',
    'animal', COMMON_PLUGINS + [
        {'manifest': PATH + 'behavior_NAFC.BehaviorManifest', 'attrs': {'NAFC': 2}},
        #{'manifest': PATH + 'stimuli.ToneInNaturalSoundsNAFC'},
        {'manifest': PATH + 'stimuli.ToneInSilenceNAFC'},
    ],
)



ParadigmDescription(
    'STD', 'Go-nogo tone detection in silence',
    'animal', COMMON_PLUGINS + [
        {'manifest': PATH + 'behavior_gonogo.BehaviorManifest'},
        {'manifest': PATH + 'stimuli.ToneInSilenceGoNogo'},
    ],
)
