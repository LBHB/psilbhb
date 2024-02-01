import matplotlib.pyplot as plt
import numpy as np
import os
import matplotlib

matplotlib.use("Qt5Agg")
from random import sample, choices
from collections import Counter
import os
import numpy as np
from psilbhb.stim.wav_set import MCWavFileSet, FgBgSet, VowelSet, CategorySet, cat_MCWavFileSets


def print_attr(fg_set):
    for dict_var in vars(fg_set):
        print(f"{dict_var}: {getattr(fg_set, dict_var)}")


def test_fgbg():
    if os.path.exists('h:/sounds'):
        soundpath_fg = 'h:/sounds/vocalizations/v1'
        soundpath_bg = 'h:/sounds/backgrounds/v1'
    else:
        soundpath_fg = '/auto/data/sounds/vocalizations/v1'
        soundpath_bg = '/auto/data/sounds/backgrounds/v1'

    vv = MCWavFileSet(
        fs=44000, path=soundpath_fg, duration=3, normalization='rms',
        fit_range=slice(0, 2), test_range=None, test_reps=2,
        channel_count=1, level=60)
    bb = MCWavFileSet(
        fs=44000, path=soundpath_bg, duration=4, normalization='rms',
        fit_range=[0, 1, 4], test_range=None, test_reps=2,
        channel_count=1, level=0)

    print(vv.names)

    w = vv.waveform(0)
    print(w.shape)

    fg_snr = [-100, -5, 0, 95, 100]

    fb = FgBgSet(FgSet=vv, BgSet=bb, fg_switch_channels=True,
                 bg_switch_channels='combinatorial',
                 combinations='all', migrate_fraction=0.5,
                 fg_snr=fg_snr, fg_delay=0.5)
    fb.update()  # not necessary but illustrative of back-end processing

    simulated_performance = [0, 0, 2, 2, 2, 1, 2, 2, 1, 2, 2, 1, 1, 2, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]

    for trial_idx, outcome in enumerate(simulated_performance):
        w = fb.trial_waveform(trial_idx)
        d = fb.trial_parameters(trial_idx)
        fb.score_response(outcome, trial_idx=trial_idx)
        print(d['trial_idx'], d['wav_set_idx'], d['fg_name'], d['fg_channel'], d['bg_name'], d['bg_channel'],
              d['response_condition'], d['current_full_rep'], d['trial_is_repeat'])

    print(f"wav_per_rep: {fb.wav_per_rep}")
    print(f"current full rep: {fb.current_full_rep}")
    print(f"scored trials: {len(fb.trial_outcomes)}")
    print(f"error trials: {sum((fb.trial_outcomes > -1) & (fb.trial_outcomes < 2))}")
    print(f"trials remaining this rep: {len(fb.trial_wav_idx) - len(fb.trial_outcomes)}")

    # plot waveforms from an example trial
    trial_idx = 0
    w = fb.trial_waveform(trial_idx)
    wb = fb.BgSet.waveform(fb.bg_index[trial_idx])
    wf = fb.FgSet.waveform(fb.fg_index[trial_idx])

    # for i in range(20):
    #    d = fb.trial_parameters(i)
    #    print(d['response_condition'])

    f, ax = plt.subplots(2, 1, sharex='col', sharey='col')
    t = np.arange(w.shape[0]) / fb.FgSet.fs
    ax[0].plot(t, w[:, 0])
    ax[0].plot(t, wb[:, 0])
    ax[0].set_title('channel 1')
    if w.shape[1] > 1:
        ax[1].plot(t, w[:, 1], label='f+b')
    if wb.shape[1] > 1:
        ax[1].plot(t, wb[:, 1], label='b')
    ax[1].legend()
    ax[1].set_title('channel 2')
    plt.tight_layout()


def test_vowels_old():
    soundpath_fg = '/auto/data/sounds/vowels/v1'

    # ['01_AE_106.wav', '02_AE_151.wav', '03_AE_201.wav',
    # '04_AW_106.wav', '05_AW_151.wav', '06_AW_201.wav',
    # '07_EE_106.wav', '08_EE_151.wav', '09_EE_201.wav',
    # '10_OO_106.wav', '11_OO_151.wav', '12_OO_201.wav',
    # 'x_silence.wav']
    fs = 44000
    vv = MCWavFileSet(
        fs=fs, path=soundpath_fg, duration=0.24, normalization='rms',
        fit_range=[0, 1, 2, 6, 7, 8, 6, 7, 8, 3, 4, 5], test_range=None, test_reps=1,
        channel_count=1, binaural_combinations='all', include_silence=False)
    vv2 = MCWavFileSet(
        fs=fs, path=soundpath_fg, duration=0.24, normalization='rms',
        fit_range=[12, 12, 12, 3, 4, 5, 12, 12, 12, 12, 12, 12], test_range=None, test_reps=1,
        channel_count=1, binaural_combinations='all', include_silence=False)
    print(vv.names)
    print(len(vv.names))

    w = vv.waveform(0)
    print(w.shape)

    fg_snr = 0

    fb = FgBgSet(FgSet=vv, BgSet=vv2, fg_switch_channels=True,
                 bg_switch_channels='combinatorial',
                 combinations='simple',
                 fg_snr=fg_snr, fg_delay=0.0)
    fb.update()  # not necessary but illustrative of back-end processing

    import sounddevice as sd
    import time
    for wav_set_idx in [1, 4, 7, 10]:
        w = fb.trial_waveform(wav_set_idx=wav_set_idx)
        d = fb.trial_parameters(wav_set_idx=wav_set_idx)
        sd.play(w, fs)
        print(d['trial_idx'], d['wav_set_idx'], d['fg_name'], d['fg_channel'], d['bg_name'], d['bg_channel'])
        time.sleep(0.4)
    simulated_performance = [0, 0, 2, 2, 2, 1, 2, 2, 1, 2, 2, 1, 1, 2, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]

    for trial_idx, outcome in enumerate(simulated_performance):
        w = fb.trial_waveform(trial_idx)
        d = fb.trial_parameters(trial_idx)
        fb.score_response(outcome, trial_idx=trial_idx)

    print(f"wav_per_rep: {fb.wav_per_rep}")
    print(f"current rep: {fb.current_repetition}")
    print(f"scored trials: {len(fb.trial_outcomes)}")
    print(f"error trials: {sum((fb.trial_outcomes > -1) & (fb.trial_outcomes < 2))}")
    print(f"trials remaining this rep: {len(fb.trial_wav_idx) - len(fb.trial_outcomes)}")

    # plot waveforms from an example trial
    trial_idx = 0
    w = fb.trial_waveform(trial_idx)
    wb = fb.BgSet.waveform(fb.bg_index[trial_idx])
    wf = fb.FgSet.waveform(fb.fg_index[trial_idx])

    f, ax = plt.subplots(2, 1, sharex='col', sharey='col')
    t = np.arange(w.shape[0]) / fb.FgSet.fs
    ax[0].plot(t, w[:, 0])
    ax[0].plot(t, wb[:, 0])
    ax[0].set_title('channel 1')
    if w.shape[1] > 1:
        ax[1].plot(t, w[:, 1], label='f+b')
    if wb.shape[1] > 1:
        ax[1].plot(t, wb[:, 1], label='b')
    ax[1].legend()
    ax[1].set_title('channel 2')
    plt.tight_layout()

    d = fb.trial_parameters(trial_idx)

    # plt.show()


def test_vowels():
    if os.path.exists('h:/sounds'):
        sound_path = 'h:/sounds/vowels/v3'
    else:
        sound_path = '/auto/data/sounds/vowels/v3'

    v = VowelSet(sound_path=sound_path, switch_channels=False,
                 target_set=[],
                 non_target_set=['OO_106', 'OO_151', 'OO_201', 'UH_106', 'UH_151', 'UH_201'],
                 repeat_count=3)
    v.update(1)
    print(v.trial_wav_idx)

    simulated_performance = [0, 0, 2, 2, 2, 1, 2, 2, 1, 2, 2, 1, 1, 2, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]

    for trial_idx, outcome in enumerate(simulated_performance):
        w = v.trial_waveform(trial_idx)
        d = v.trial_parameters(trial_idx)
        v.score_response(outcome, trial_idx=trial_idx)

        print(d['s1_name'] + ' ' + d['s2_name'])
    print('done')
    # import sounddevice as sd
    # import time
    # fs=v.wavset.fs
    # for wav_set_idx in [1,7,10, 15, 20]:
    #    w=v.trial_waveform(wav_set_idx=wav_set_idx)
    #    d=v.trial_parameters(wav_set_idx=wav_set_idx)
    #    sd.play(w.T,fs)
    #    print(d['trial_idx'], d['wav_set_idx'], d['s1_name'], d['s2_name'])
    #    time.sleep(1)


def test_vowels2():
    if os.path.exists('h:/sounds'):
        soundpath_fg = 'h:/sounds/vocalizations/v1'
        soundpath_bg = 'h:/sounds/backgrounds/v1'
    else:
        soundpath_fg = '/auto/data/sounds/vocalizations/v1'
        soundpath_bg = '/auto/data/sounds/backgrounds/v1'

    vv = MCWavFileSet(
        fs=44000, path=soundpath_fg, duration=3, normalization='rms',
        fit_range=slice(0, 1), test_range=None, test_reps=2,
        channel_count=1, level=60)
    bb = MCWavFileSet(
        fs=44000, path=soundpath_bg, duration=4, normalization='rms',
        fit_range=[0, 4], test_range=None, test_reps=2,
        channel_count=1, level=0)

    print(vv.names)

    w = vv.waveform(0)
    print(w.shape)

    fg_snr = -5

    fb = FgBgSet(FgSet=vv, BgSet=bb, fg_switch_channels=True,
                 bg_switch_channels='combinatorial',
                 combinations='all', migrate_fraction=0.5,
                 fg_snr=fg_snr, fg_delay=0.5)
    fb.update()  # not necessary but illustrative of back-end processing

    simulated_performance = [0, 0, 2, 2, 2, 1, 2, 2, 1, 2, 2, 1, 1, 2, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]

    for trial_idx, outcome in enumerate(simulated_performance):
        w = fb.trial_waveform(trial_idx)
        d = fb.trial_parameters(trial_idx)
        fb.score_response(outcome, trial_idx=trial_idx)
        print(d['trial_idx'], d['wav_set_idx'], d['fg_name'], d['fg_channel'], d['bg_name'], d['bg_channel'],
              d['response_condition'], d['current_full_rep'], d['trial_is_repeat'])

    print(f"wav_per_rep: {fb.wav_per_rep}")
    print(f"current full rep: {fb.current_full_rep}")
    print(f"scored trials: {len(fb.trial_outcomes)}")
    print(f"error trials: {sum((fb.trial_outcomes > -1) & (fb.trial_outcomes < 2))}")
    print(f"trials remaining this rep: {len(fb.trial_wav_idx) - len(fb.trial_outcomes)}")

    # plot waveforms from an example trial
    trial_idx = 5
    w = fb.trial_waveform(trial_idx)
    wb = fb.BgSet.waveform(fb.bg_index[trial_idx]).T
    wf = fb.FgSet.waveform(fb.fg_index[trial_idx]).T

    # for i in range(20):
    #    d = fb.trial_parameters(i)
    #    print(d['response_condition'])

    f, ax = plt.subplots(2, 1, sharex='col', sharey='col')
    t = np.arange(w.shape[1]) / fb.FgSet.fs
    ax[0].plot(t, w[0, :])
    ax[0].plot(t, wb[0, :])
    ax[0].set_title('channel 1')
    if w.shape[0] > 1:
        ax[1].plot(t, w[1, :], label='f+b')
    if wb.shape[0] > 1:
        ax[1].plot(t, wb[1, :], label='b')
    ax[1].legend()
    ax[1].set_title('channel 2')
    plt.tight_layout()


def test_categories_using_VowelSet():
    if os.path.exists('h:/sounds'):
        sound_path = 'h:/sounds/vocalizations/v3_vocoding'
    else:
        sound_path = '/auto/data/sounds/vocalizations/v3_vocoding'

    all_ferret_files = [file for file in os.listdir(sound_path) if file.endswith(".wav") and file.startswith("fer")]
    all_speech_files = [file for file in os.listdir(sound_path) if file.endswith(".wav") and file.startswith("spe")]
    all_catch_files = [file for file in os.listdir(sound_path) if file.endswith(".wav") and file.startswith("ENV")]

    num_reg_files = 3
    num_catch_files_per_env = 3

    ferret_slice = slice(0, num_reg_files)
    speech_slice = ferret_slice

    sliced_ferret_files = all_ferret_files[ferret_slice]
    sliced_speech_files = all_speech_files[speech_slice]

    regular_stims = [x[:-4] + '+' + y for x in sliced_ferret_files for y in sliced_speech_files] + \
                    [x[:-4] + '+' + y for x in sliced_speech_files for y in sliced_ferret_files]

    # initialize catch_slice based on ferret vocals - we don't want the same ferret vocal and vocoded vocal played together
    catch_slice = [fi for fi in range(len(all_ferret_files)) if fi not in range(ferret_slice.start, ferret_slice.stop)]
    catch_slice = sample(catch_slice, num_catch_files_per_env)
    valid_catch_files = [all_ferret_files[idx] for idx in catch_slice]
    env_bands = [2, 8, 32]
    catch_stims = ['ENV' + str(nband) + '_' + file for nband in env_bands for file in valid_catch_files]
    catch_stims = [x[:-4] + '+' + y for x in sliced_ferret_files for y in catch_stims] + \
                  [x[:-4] + '+' + y for x in catch_stims for y in sliced_ferret_files]

    v_vowel = VowelSet(sound_path=sound_path, switch_channels=False, duration=0.24,
                       target_set=regular_stims, non_target_set=[], catch_set=catch_stims, repeat_count=3)

    v_vowel.update(1)
    print(v_vowel.trial_wav_idx)
    # print all stim being played
    # [print(v_vowel.wavset.names[v_vowel.stim1idx[idx]]) for idx in v_vowel.trial_wav_idx]

    # simulated_performance = [0, 0, 3, 3, 2, 2, 2, 1, 2, 2, 1, 2, 2, 1, 1, 2, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 1, 1, 2]
    simulated_performance = [2 for _ in range(200)]

    for trial_idx, outcome in enumerate(simulated_performance):
        w = v_vowel.trial_waveform(trial_idx)
        d = v_vowel.trial_parameters(trial_idx)
        v_vowel.score_response(outcome, trial_idx=trial_idx)
        print('vow: < ' + d['s1_name'] + ' & ' + d['s2_name'] + ' >')

    print('done')


def test_categories():
    if os.path.exists('h:/sounds'):
        soundpath_fg = 'h:/sounds/Categories/v3_vocoding'
        soundpath_bg = 'h:/sounds/Categories/speech_stims'
        soundpath_catch_bg = 'h:/sounds/Categories/chimeric_voc'
    else:
        soundpath_fg = '/auto/users/satya/code/projects_getting_started/explore_bignat/ferret_vocals/v3_vocoding'
        soundpath_bg = '/auto/users/satya/code/projects_getting_started/explore_bignat/ferret_vocals/speech_stims'
        soundpath_catch_bg = '/auto/users/satya/code/projects_getting_started/explore_bignat/ferret_vocals/chimeric_voc'

    fg_set = MCWavFileSet(
        fs=44000, path=soundpath_fg, duration=3, normalization='rms',
        fit_range=slice(0, 2), test_range=None, test_reps=2,
        channel_count=1, level=60)
    bg_set = MCWavFileSet(
        fs=44000, path=soundpath_bg, duration=3, normalization='rms',
        fit_range=[0, 1, 4], test_range=None, test_reps=2,
        channel_count=1, level=60)
    catch_id_range = [0, 1, 4]
    catch_bg_set = MCWavFileSet(filelabels='C',
                                fs=44000, path=soundpath_catch_bg, duration=3, normalization='rms',
                                fit_range=catch_id_range, test_range=None, test_reps=2,
                                channel_count=1, level=60)

    print(fg_set.names)
    w = fg_set.waveform(0)
    print(w.shape)

    fg_snr = [0, ]

    # cat_bg_set = cat_MCWavFileSets(bg_set, catch_bg_set, frac_set1=.8)
    # fb = FgBgSet(FgSet=fg_set, BgSet=cat_bg_set, fg_switch_channels=True, bg_switch_channels='opposite',
    #                  combinations='all', fg_snr=fg_snr)
    fb = CategorySet(FgSet=fg_set, BgSet=bg_set, CatchBGSet=catch_bg_set, CatchBG_frac=.8,
                     fg_switch_channels=True, bg_switch_channels='opposite',
                     combinations='all', fg_snr=fg_snr)
    fb.update()  # not necessary but illustrative of back-end processing

    simulated_performance = [0, 0, 2, 2, 2, 1, 2, 2, 1, 2, 2, 1, 1, 2, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
    # simulated_performance = [2 for _ in range(200)]

    played_fg = [[] for _ in simulated_performance]
    played_bg = [[] for _ in simulated_performance]

    for trial_idx, outcome in enumerate(simulated_performance):
        w = fb.trial_waveform(trial_idx)
        d = fb.trial_parameters(trial_idx)
        fb.score_response(outcome, trial_idx=trial_idx)
        print(d['trial_idx'], d['wav_set_idx'], d['fg_name'], d['fg_channel'], d['bg_name'], d['bg_channel'],
              d['response_condition'], d['current_full_rep'], d['trial_is_repeat'])
        played_fg[trial_idx] = d['fg_name']
        played_bg[trial_idx] = d['bg_name']

    print('FG')
    print(Counter(played_fg).keys())  # equals to list(set(words))
    print(Counter(played_fg).values())  # counts the elements' frequency

    print('BG')
    print(Counter(played_bg).keys())  # equals to list(set(words))
    print(Counter(played_bg).values())  # counts the elements' frequency
    print('----------------------------------------------------------------------------------------------------')

    print(f"wav_per_rep: {fb.wav_per_rep}")
    print(f"current full rep: {fb.current_full_rep}")
    print(f"scored trials: {len(fb.trial_outcomes)}")
    print(f"error trials: {sum((fb.trial_outcomes > -1) & (fb.trial_outcomes < 2))}")
    print(f"trials remaining this rep: {len(fb.trial_wav_idx) - len(fb.trial_outcomes)}")

    # plot waveforms from an example trial
    trial_idx = 0
    w = fb.trial_waveform(trial_idx).T
    wb = fb.BgSet.waveform(fb.bg_index[trial_idx])
    wf = fb.FgSet.waveform(fb.fg_index[trial_idx])

    # for i in range(20):
    #    d = fb.trial_parameters(i)
    #    print(d['response_condition'])

    f, ax = plt.subplots(2, 1, sharex='col', sharey='col')
    t = np.arange(w.shape[0]) / fb.FgSet.fs
    ax[0].plot(t, w[:, 0])
    # ax[0].plot(t,wb[:,0])
    ax[0].set_title('channel 1')
    if w.shape[1] > 1:
        ax[1].plot(t, w[:, 1], label='f+b')
    if wb.shape[1] > 1:
        ax[1].plot(t, wb[:, 1], label='b')
    ax[1].legend()
    ax[1].set_title('channel 2')
    plt.tight_layout()


def test_simple_category():
    if os.path.exists('h:/sounds'):
        soundpath_fg = 'h:/sounds/Categories/v3_vocoding'
        soundpath_bg = 'h:/sounds/Categories/speech_stims'
        soundpath_catch_bg = 'h:/sounds/Categories/chimeric_voc'
        soundpath_OAnoise = 'h:/sounds/Categories/noise_vocPSDmatched'
    else:
        soundpath_fg = '/auto/users/satya/code/projects_getting_started/explore_bignat/ferret_vocals/v3_vocoding'
        soundpath_bg = '/auto/users/satya/code/projects_getting_started/explore_bignat/ferret_vocals/speech_stims'
        soundpath_catch_bg = '/auto/users/satya/code/projects_getting_started/explore_bignat/ferret_vocals/chimeric_voc'
        soundpath_OAnoise = '/auto/users/satya/code/projects_getting_started/explore_bignat/ferret_vocals/noise_vocPSDmatched'

    FgSet = MCWavFileSet(fs=44000, path=soundpath_fg, duration=3, normalization='rms',
                         fit_range=-1, channel_count=1, level=55)
    BgSet = MCWavFileSet(fs=44000, path=soundpath_bg, duration=3, normalization='rms',
                         fit_range=-1, channel_count=1, level=55)
    CatchBgSet = MCWavFileSet(fs=44000, path=soundpath_catch_bg, duration=3, normalization='rms',
                              fit_range=-1, channel_count=1, level=55)
    OAnoiseSet = MCWavFileSet(fs=44000, path=soundpath_OAnoise, duration=3, normalization='rms',
                              fit_range=-1, channel_count=1, level=55)

    print(FgSet.names)
    w = FgSet.waveform(0)
    print(w.shape)

    # catch_ferret_id = 3
    # n_env_bands = [2, 8, 32]
    # reg2catch_ratio= 6 # 6 for 85% regular trials, 7 for 87.5% regular trials: rest catch trials

    fg_snr = [0., ]

    # fb = CategorySet(FgSet=FgSet, BgSet=BgSet, CatchBgSet=CatchBgSet, OAnoiseSet=OAnoiseSet, fg_switch_channels=True,
    #                  bg_switch_channels='opposite', combinations='custom',
    #                  fg_snr=fg_snr, catch_ferret_id=3, n_env_bands=[2, 8, 32], reg2catch_ratio=6, unique_overall_SNR= [np.inf])

    fb = CategorySet(FgSet=FgSet, BgSet=BgSet, CatchBgSet=CatchBgSet, OAnoiseSet=OAnoiseSet, fg_switch_channels=True,
                     bg_switch_channels='opposite', combinations='custom',
                     fg_snr=fg_snr, catch_ferret_id=3, n_env_bands=[2, 8, 32], reg2catch_ratio=6,
                     unique_overall_SNR=[-10, 0, 10])

    fb.update()  # not necessary but illustrative of back-end processing

    simulated_performance = [0, 2, 2, 1, 2, 2, 0, 2, 0, 0, 0, 2, 2, 2, 0, 1, 2, 2, 2, 2, 0, 0,
                             2, 2, 2, 2, 0, 2, 0, 2, 2, 2, 2, 2, 0, 2, 1, 2, 2, 0, 2, 2, 0, 2,
                             2, 2, 0, 2, 2, 0, 1, 2, 2, 0, 2, 0, 2, 2, 2, 2, 0, 2, 2, 2, 0, 2,
                             2, 2, 0, 2, 1, 2, 1, 2, 1, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 0, 2,
                             2, 0, 2, 2, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 0, 2, 1, 1,
                             2, 1, 2, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2,
                             2, 2, 2, 0, 2, 2, 0, 1, 2, 2, 1, 2, 2, 2, 2, 0, 2, 2, 1, 2, 2, 2,
                             1, 2, 1, 0, 1, 2, 2, 1, 2, 2, 2, 2, 2, 2, 1]
    # simulated_performance = [2 for _ in range(200)]

    played_fg = [[] for _ in simulated_performance]
    played_bg = [[] for _ in simulated_performance]
    played_overall_snr = [[] for _ in simulated_performance]

    for trial_idx, outcome in enumerate(simulated_performance):
        w = fb.trial_waveform(trial_idx)
        d = fb.trial_parameters(trial_idx)
        fb.score_response(outcome, trial_idx=trial_idx)
        print(d['trial_idx'], d['wav_set_idx'], d['overall_snr'], d['fg_name'], d['fg_channel'], d['bg_name'],
              d['bg_channel'],
              d['response_condition'], d['current_full_rep'], d['trial_is_repeat'])
        played_fg[trial_idx] = d['fg_name']
        played_bg[trial_idx] = d['bg_name']
        played_overall_snr[trial_idx] = d['overall_snr']

    print('FG')
    print(Counter(played_fg).keys())  # equals to list(set(words))
    print(Counter(played_fg).values())  # counts the elements' frequency

    print('BG')
    print(Counter(played_bg).keys())  # equals to list(set(words))
    print(Counter(played_bg).values())  # counts the elements' frequency
    print('----------------------------------------------------------------------------------------------------')

    print(f"wav_per_rep: {fb.wav_per_rep}")
    print(f"current full rep: {fb.current_full_rep}")
    print(f"scored trials: {len(fb.trial_outcomes)}")
    print(f"error trials: {sum((fb.trial_outcomes > -1) & (fb.trial_outcomes < 2))}")
    print(f"trials remaining this rep: {len(fb.trial_wav_idx) - len(fb.trial_outcomes)}")

    print("SNR statistics")
    print(Counter(played_overall_snr).keys())
    print(Counter(played_overall_snr).values())  # counts the elements' frequency

    # plot waveforms from an example trial
    # trial_idx = 0
    for trial_idx in np.array([0, 1]):
        w = fb.trial_waveform(trial_idx).T
        wb = fb.BgSet.waveform(fb.bg_index[trial_idx])

        f, ax = plt.subplots(2, 1, sharex='col', sharey='col')
        t = np.arange(w.shape[0]) / fb.FgSet.fs
        ax[0].plot(t, w[:, 0])
        # ax[0].plot(t,wb[:,0])
        ax[0].set_title('channel 1')
        if w.shape[1] > 1:
            ax[1].plot(t, w[:, 1], label='f')
        if wb.shape[1] > 1:
            ax[1].plot(t, wb[:, 1], label='b')
        ax[1].legend()
        ax[1].set_title('channel 2')
        plt.tight_layout()
        plt.show()


# test_fgbg()
# test_vowels()
# test_vowels2()
# test_categories_using_VowelSet()
# print('wth')
# test_categories()
test_simple_category()
