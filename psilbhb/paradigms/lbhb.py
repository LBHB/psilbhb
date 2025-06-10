from psi.experiment.api import ParadigmDescription


PATH = 'psilbhb.paradigms.behavior.'
CORE_PATH = 'psi.paradigms.core.'


COMMON_PLUGINS = [
    {'manifest': 'psilbhb.paradigms.video.PSIVideo',
     'attrs': {
         'id': 'psivideo',
         'title': 'Video (top)',
         'port': 33331,
         'filename': 'recording.avi',
     }},
    {'manifest': 'psilbhb.paradigms.video.PSIVideo',
     'attrs': {
         'id': 'psivideo_side',
         'title': 'Video (side)',
         'port': 33332,
         'filename': 'video_side.avi',
     }},
    {'manifest': 'psilbhb.paradigms.video.PSIVideo',
     'attrs': {
         'id': 'psivideo_side2',
         'title': 'Video (side2)',
         'port': 33333,
         'filename': 'video_side2.avi',
     }},
    {'manifest': 'psilbhb.paradigms.video.PSIVideo',
     'attrs': {
         'id': 'psivideo_pupil',
         'title': 'Video (pupil)',
         'port': 33334,
         'filename': 'video_pupil.avi',
     }},
    {'manifest': 'psilbhb.paradigms.video.PSIVideo',
     'attrs': {
         'id': 'psivideo_side3',
         'title': 'Video (side3)',
         'port': 33335,
         'filename': 'video_pupil.avi',
     }},
    {'manifest': 'psilbhb.paradigms.openephys.OpenEphysManifest'},
    {'manifest': PATH + 'behavior_mixins.SignalFFTViewManifest',
     'attrs': {
         'fft_time_span': 1,
         'fft_freq_lb': 5,
         'fft_freq_ub': 24000,
         'y_label': 'Level (dB)'},
     },
    {'manifest': 'psi.paradigms.core.signal_mixins.SignalFFTViewManifest',
     'attrs': {
         'id': 'microphone_1_fft',
         'title': 'Microphone 1 PSD',
         'fft_time_span': 1,
         'fft_freq_lb': 5,
         'fft_freq_ub': 24000,
         'y_label': 'Level (dB)',
         'apply_calibration': True,
         'source_name': 'microphone_1',
         },
     },
    {'manifest': 'psi.paradigms.core.signal_mixins.SignalFFTViewManifest',
     'attrs': {
         'id': 'microphone_2_fft',
         'title': 'Microphone 2 PSD',
         'fft_time_span': 1,
         'fft_freq_lb': 5,
         'fft_freq_ub': 24000,
         'y_label': 'Level (dB)',
         'apply_calibration': True,
         'source_name': 'microphone_2',
         },
     },
]


ParadigmDescription(
    'NFB-passive', '(NFB) Passive FG in natural background',
    'animal', COMMON_PLUGINS + [
        {'manifest': PATH + 'passive.PassiveManifest',},
        {'manifest': PATH + 'wav_set_manifest.WavSetManifest', 'required': True, 'attrs': {'stim_class_name': 'FgBgSet'}},
    ],
)


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
    'NTD-passive', '(NTD) Tone in natural background (passive)',
    'animal', COMMON_PLUGINS + [
        {'manifest': PATH + 'behavior_mixins.BaseGoNogoMixin'},
        {'manifest': PATH + 'behavior_mixins.WaterBolusDispenser',
         'attrs': {'output_name': 'water_dispense_1',
                   'event_name': 'deliver_reward'}},
        {'manifest': PATH + 'behavior_gonogo.AutoBehaviorManifest'},
        {'manifest': PATH + 'stimuli.ToneInNaturalSoundsGoNogo'},
    ],
)

ParadigmDescription(
    'NFB', '(NFB) Two AFC foreground detection in natural background',
    'animal', COMMON_PLUGINS + [
        {'manifest': PATH + 'behavior_nafc.BehaviorManifest',
         'attrs': {'N_response': 2}},
        {'manifest': PATH + 'wav_set_manifest.WavSetManifest', 'required': True,
         'attrs': {'stim_class_name': 'FgBgSet'}
         },
        {'manifest': PATH + 'behavior_mixins.WaterBolusDispenser',
         'attrs': {'output_name': 'water_dispense_1',
                   'event_name': 'deliver_reward_1'}},
        {'manifest': PATH + 'behavior_mixins.WaterBolusDispenser',
         'attrs': {'output_name': 'water_dispense_2',
                   'event_name': 'deliver_reward_2'}},
    ],
)


ParadigmDescription(
    'NFC', '(NFC) Two AFC foreground detection in continuous natural background',
    'animal', COMMON_PLUGINS + [
        {'manifest': PATH + 'behavior_nafc.BehaviorManifest',
         'attrs': {'N_response': 2, 'N_output': 2}},
        {'manifest': PATH + 'wav_set_manifest.WavSetManifest', 'required': True,
         'attrs': {'stim_class_name': 'FgBgSet'}
         },
        {'manifest': PATH + 'wav_set_manifest.WavSetManifest', 'required': True,
         'attrs': {'stim_class_name': 'BigNat', 'output_type': 'continuous', 'prefix': 'background_'}
         },
        #{'manifest': PATH + 'wav_set_manifest.WavSetManifest', 'required': True,
        # 'attrs': {'stim_class_name': 'Silence', 'output_type': 'continuous', 'prefix': 'background_'}
        # },
        {'manifest': PATH + 'behavior_mixins.WaterBolusDispenser',
         'attrs': {'output_name': 'water_dispense_1',
                   'event_name': 'deliver_reward_1'}},
        {'manifest': PATH + 'behavior_mixins.WaterBolusDispenser',
         'attrs': {'output_name': 'water_dispense_2',
                   'event_name': 'deliver_reward_2'}},
    ],
)


ParadigmDescription(
    'VOW', '(VOW) Vowel discrimination',
    'animal', COMMON_PLUGINS + [
        {'manifest': PATH + 'behavior_nafc.BehaviorManifest',
         'attrs': {'N_response': 2}},
        {'manifest': PATH + 'wav_set_manifest.WavSetManifest', 'required': True,
         'attrs': {'stim_class_name': 'VowelSet'}},
        {'manifest': PATH + 'behavior_mixins.WaterBolusDispenser',
         'attrs': {'output_name': 'water_dispense_1',
                   'event_name': 'deliver_reward_1'}},
        {'manifest': PATH + 'behavior_mixins.WaterBolusDispenser',
         'attrs': {'output_name': 'water_dispense_2',
                   'event_name': 'deliver_reward_2'}},
    ],
)


ParadigmDescription(
    'VGN', '(VGN) Go/nogo Vowel discrimination',
    'animal', COMMON_PLUGINS + [
        {'manifest': PATH + 'behavior_nafc.BehaviorManifest',
         'attrs': {'N_response': 1}},
        {'manifest': PATH + 'wav_set_manifest.WavSetManifest', 'required': True,
         'attrs': {'stim_class_name': 'VowelSet'}},
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
        {'manifest': PATH + 'behavior_nafc.BehaviorManifest',
         'attrs': {'N_response': 2}},
        {'manifest': PATH + 'wav_set_manifest.CategorySetManifest', 'required': True},
        {'manifest': PATH + 'behavior_mixins.WaterBolusDispenser',
         'attrs': {'output_name': 'water_dispense_1',
                   'event_name': 'deliver_reward_1'}},
        {'manifest': PATH + 'behavior_mixins.WaterBolusDispenser',
         'attrs': {'output_name': 'water_dispense_2',
                   'event_name': 'deliver_reward_2'}},
    ],
)

ParadigmDescription(
    'AMF', '(AMF) Amplitude modulation/fusion 2AFC',
    'animal', COMMON_PLUGINS + [
        {'manifest': PATH + 'behavior_nafc.BehaviorManifest',
         'attrs': {'N_response': 2}},
        {'manifest': PATH + 'wav_set_manifest.WavSetManifest', 'required': True,
         'attrs': {'stim_class_name': 'AMFusion'}
         },
        {'manifest': PATH + 'behavior_mixins.WaterBolusDispenser',
         'attrs': {'output_name': 'water_dispense_1',
                   'event_name': 'deliver_reward_1'}},
        {'manifest': PATH + 'behavior_mixins.WaterBolusDispenser',
         'attrs': {'output_name': 'water_dispense_2',
                   'event_name': 'deliver_reward_2'}},
    ],
)

ParadigmDescription(
    'OLP', 'OLP - Overlapping Sounds Passive [Placeholder]',
    'animal', COMMON_PLUGINS + [
        {'manifest': PATH + 'wav_set_manifest.WavSetManifest', 'required': True,
         'attrs': {'stim_class_name': 'OverlappingSounds'}
         },
    ],
)

ParadigmDescription(
    'FTC', '(FTC) Passive random tone',
    'animal', COMMON_PLUGINS + [
        {'manifest': PATH + 'passive.PassiveManifest',},
        {'manifest': PATH + 'wav_set_manifest.WavSetManifest', 'required': True, 'attrs': {'stim_class_name': 'RandomTone'}},
    ],
)

ParadigmDescription(
    'BNB', '(BNB) Bandpass noise burst',
    'animal', COMMON_PLUGINS + [
        {'manifest': PATH + 'passive.PassiveManifest',},
        {'manifest': PATH + 'wav_set_manifest.WavSetManifest', 'required': True, 'attrs': {'stim_class_name': 'BandpassNoise'}},
    ],
)

ParadigmDescription(
    'BLT', '(BLT) Passive binaural level tuning',
    'animal', COMMON_PLUGINS + [
        {'manifest': PATH + 'passive.PassiveManifest',},
        {'manifest': PATH + 'wav_set_manifest.WavSetManifest', 'required': True, 'attrs': {'stim_class_name': 'BinauralTone'}},
    ],
)

ParadigmDescription(
    'BAM', '(BAM) Binaural AM tones',
    'animal', COMMON_PLUGINS + [
        {'manifest': PATH + 'passive.PassiveManifest',},
        {'manifest': PATH + 'wav_set_manifest.WavSetManifest', 'required': True, 'attrs': {'stim_class_name': 'BinauralAM'}},
    ],
)

ParadigmDescription(
    'BVP', '(BVP) Vowel passive',
    'animal', COMMON_PLUGINS + [
        {'manifest': PATH + 'passive.PassiveManifest', },
        {'manifest': PATH + 'wav_set_manifest.WavSetManifest', 'required': True,
         'attrs': {'stim_class_name': 'VowelSet'}},
    ],
)

ParadigmDescription(
    'BNT', '(BNT) Big natural sound set',
    'animal', COMMON_PLUGINS + [
        {'manifest': PATH + 'passive.PassiveManifest', },
        {'manifest': PATH + 'wav_set_manifest.WavSetManifest', 'required': True,
         'attrs': {'stim_class_name': 'BigNat'}},
    ],
)
