from psi.experiment.api import ParadigmDescription


PATH = 'psilbhb.paradigms.behavior.'
CORE_PATH = 'psi.paradigms.core.'


COMMON_PLUGINS = [
    {'manifest': 'psilbhb.paradigms.video.PSIVideo'},
    {'manifest': 'psilbhb.paradigms.openephys.OpenEphysManifest'},
        {'manifest': PATH + 'behavior_mixins.SignalFFTViewManifest',
        'attrs': {'fft_time_span': 1, 'fft_freq_lb': 5, 'fft_freq_ub': 24000,
                'y_label': 'Level (dB)'}
            },
]



ParadigmDescription(
    'NTD-gonogo-np', '(NTD) Go-nogo tone detection in natural background (initiated)',
    'animal', COMMON_PLUGINS + [
        {'manifest': PATH + 'behavior_mixins.BaseGoNogoMixin'},
        {'manifest': PATH + 'behavior_mixins.WaterBolusDispenser',
         'attrs': {'output_name': 'water_dispense_1',
                   'event_name': 'deliver_reward'}},
        {'manifest': PATH + 'behavior_gonogo.InitiatedBehaviorManifest'},
        {'manifest': PATH + 'stimuli.ToneInNaturalSoundsGoNogo'},
    ],
)

ParadigmDescription(
    'NFB', '(NFB) Two AFC foreground detection in natural background',
    'animal', COMMON_PLUGINS + [
        {'manifest': PATH + 'behavior_2afc.BehaviorManifest'},
        {'manifest': PATH + 'wav_set_manifest.FgBgSetManifest', 'required': True},
        {'manifest': PATH + 'behavior_mixins.WaterBolusDispenser',
         'attrs': {'output_name': 'water_dispense_1',
                   'event_name': 'deliver_reward_1'}},
        {'manifest': PATH + 'behavior_mixins.WaterBolusDispenser',
         'attrs': {'output_name': 'water_dispense_2',
                   'event_name': 'deliver_reward_2',
                   'add_params': False}},
    ],
)


ParadigmDescription(
    'VOW', '(VOW) Vowel discrimination',
    'animal', COMMON_PLUGINS + [
        {'manifest': PATH + 'behavior_2afc.BehaviorManifest'},
        {'manifest': PATH + 'wav_set_manifest.VowelSetManifest', 'required': True},
        {'manifest': PATH + 'behavior_mixins.WaterBolusDispenser',
         'attrs': {'output_name': 'water_dispense_1',
                   'event_name': 'deliver_reward_1'}},
        {'manifest': PATH + 'behavior_mixins.WaterBolusDispenser',
         'attrs': {'output_name': 'water_dispense_2',
                   'event_name': 'deliver_reward_2',
                   'add_params': False}},
    ],
)


ParadigmDescription(
    'VOW', '(VOWN) Go/nogo Vowel discrimination',
    'animal', COMMON_PLUGINS + [
        {'manifest': PATH + 'behavior_nafc.BehaviorManifest'},
        {'manifest': PATH + 'wav_set_manifest.VowelSetManifest', 'required': True},
        {'manifest': PATH + 'behavior_mixins.WaterBolusDispenser',
         'attrs': {'output_name': 'water_dispense_1',
                   'event_name': 'deliver_reward_1'}},
    ],
)


ParadigmDescription(
    'STD', '(STD) Go-nogo tone detection in silence',
    'animal', COMMON_PLUGINS + [
        {'manifest': PATH + 'behavior_gonogo.BehaviorManifest'},
        {'manifest': PATH + 'stimuli.ToneInSilenceGoNogo'},
        {'manifest': PATH + 'behavior_mixins.WaterBolusDispenser'},
    ],
)


ParadigmDescription(
    'CAT', '(CAT) Natural category discrimination',
    'animal', COMMON_PLUGINS + [
        {'manifest': PATH + 'behavior_2afc.BehaviorManifest'},
        {'manifest': PATH + 'wav_set_manifest.CategorySetManifest', 'required': True},
        {'manifest': PATH + 'behavior_mixins.WaterBolusDispenser',
         'attrs': {'output_name': 'water_dispense_1',
                   'event_name': 'deliver_reward_1'}},
        {'manifest': PATH + 'behavior_mixins.WaterBolusDispenser',
         'attrs': {'output_name': 'water_dispense_2',
                   'event_name': 'deliver_reward_2',
                   'add_params': False}},
    ],
)

