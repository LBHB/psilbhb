import matplotlib.pyplot as plt
import numpy as np
import os
import matplotlib
matplotlib.use("Qt5Agg")

from random import sample, choices
from collections import Counter
import os
import numpy as np
from psilbhb.stim.wav_set import MCWavFileSet, FgBgSet, VowelSet, OldCategorySet, CategorySet


def print_attr(fg_set):
    for dict_var in vars(fg_set):
        print(f"{dict_var}: {getattr(fg_set, dict_var)}")


def test_old_simple_category():
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

    fs = 44000
    FgSet = MCWavFileSet(fs=fs, path=soundpath_fg, duration=3, normalization='rms',
                         fit_range=-1, channel_count=1, level=55)
    BgSet = MCWavFileSet(fs=fs, path=soundpath_bg, duration=3, normalization='rms',
                         fit_range=-1, channel_count=1, level=55)
    CatchBgSet = MCWavFileSet(fs=fs, path=soundpath_catch_bg, duration=3, normalization='rms',
                              fit_range=-1, channel_count=1, level=55)
    OAnoiseSet = MCWavFileSet(fs=fs, path=soundpath_OAnoise, duration=3, normalization='rms',
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

    fb = OldCategorySet(FgSet=FgSet, BgSet=BgSet, CatchBgSet=CatchBgSet, OAnoiseSet=OAnoiseSet,
                     fg_switch_channels=True, bg_switch_channels='opposite', combinations='custom',
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
    # simulated_performance = [2 for _ in range(84)]
    simulated_performance = [2 for _ in range(200)]

    played_fg = [[] for _ in simulated_performance]
    played_bg = [[] for _ in simulated_performance]
    overall_snr = [[] for _ in simulated_performance]

    for trial_idx_py, outcome in enumerate(simulated_performance):
        trial_idx = trial_idx_py + 1
        w = fb.trial_waveform(trial_idx)
        d = fb.trial_parameters(trial_idx)
        fb.score_response(outcome, trial_idx=trial_idx)
        print(d['trial_idx'], d['wav_set_idx'], d['overall_snr'], d['fg_name'], d['fg_channel'], d['bg_name'],
              d['bg_channel'], d['response_condition'], d['current_full_rep'], d['trial_is_repeat'])
        played_fg[trial_idx_py] = d['fg_name']
        played_bg[trial_idx_py] = d['bg_name']
        overall_snr[trial_idx_py] = d['overall_snr']

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
    print(Counter(overall_snr).keys())
    print(Counter(overall_snr).values())  # counts the elements' frequency

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

def test_CategorySet():
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


    # fb = CategorySet(FgSet=FgSet, BgSet=BgSet, CatchBgSet=CatchBgSet, OAnoiseSet=OAnoiseSet, fg_switch_channels=True,
    #                  bg_switch_channels='opposite', combinations='custom',
    #                  fg_snr=fg_snr, catch_ferret_id=3, n_env_bands=[2, 8, 32], reg2catch_ratio=6,
    #                  unique_overall_SNR= [np.inf])

    params = CategorySet.default_values()
    # params.update(dict(fg_path=soundpath_fg, bg_path=soundpath_bg, catch_bg_path=soundpath_catch_bg,
    #                    fg_range=-1, bg_range=-1, catch_bg_range=-1, duration=3.0, normalization='rms',
    #                    fg_switch_channels=True, contra_n=1, ipsi_n=0, diotic_n=0, combinations='custom',
    #                    fg_delay=0., fg_level=55, bg_level=55, random_seed=1, catch_ferret_id=3,
    #                    n_env_bands=[2, 8, 32], reg2catch_ratio=6, OAnoise_SNR=[np.inf]))
    # params.update(dict(fg_path=soundpath_fg, bg_path=soundpath_bg, catch_bg_path='',
    #                    OAnoise_path=soundpath_OAnoise, fg_range=-1, bg_range=-1, duration=3.0, normalization='rms',
    #                    fg_switch_channels=True, contra_n=1, ipsi_n=0, diotic_n=0, combinations='all',
    #                    fg_delay=0., fg_level=55, bg_level=55, random_seed=1, OAnoise_SNR=[np.inf, 10, 5, 0]))

    # This is just the clean condition
    params.update(dict(fg_path=soundpath_fg, bg_path=soundpath_bg, catch_bg_path='',
                       OAnoise_path='', fg_range=-1, bg_range=-1, duration=3.0, normalization='rms',
                       fg_switch_channels=True, contra_n=1, ipsi_n=0, diotic_n=0, combinations='all',
                       fg_delay=0., fg_level=55, bg_level=55, random_seed=1, OAnoise_SNR=[np.inf]))

    fb = CategorySet(**params)
    fb.update()  # not necessary but illustrative of back-end processing

    # simulated_performance = [0, 2, 2, 1, 2, 2, 0, 2, 0, 0, 0, 2, 2, 2, 0, 1, 2, 2, 2, 2, 0, 0,
    #                          2, 2, 2, 2, 0, 2, 0, 2, 2, 2, 2, 2, 0, 2, 1, 2, 2, 0, 2, 2, 0, 2,
    #                          2, 2, 0, 2, 2, 0, 1, 2, 2, 0, 2, 0, 2, 2, 2, 2, 0, 2, 2, 2, 0, 2,
    #                          2, 2, 0, 2, 1, 2, 1, 2, 1, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 0, 2,
    #                          2, 0, 2, 2, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 0, 2, 1, 1,
    #                          2, 1, 2, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2,
    #                          2, 2, 2, 0, 2, 2, 0, 1, 2, 2, 1, 2, 2, 2, 2, 0, 2, 2, 1, 2, 2, 2,
    #                          1, 2, 1, 0, 1, 2, 2, 1, 2, 2, 2, 2, 2, 2, 1]
    simulated_performance = [2 for _ in range(200)]

    played_fg = [[] for _ in simulated_performance]
    played_bg = [[] for _ in simulated_performance]
    overall_snr = [[] for _ in simulated_performance]

    for trial_idx_py, outcome in enumerate(simulated_performance):
        trial_idx = trial_idx_py + 1
        w = fb.trial_waveform(trial_idx)
        d = fb.trial_parameters(trial_idx)
        fb.score_response(outcome, trial_idx=trial_idx)
        print(outcome, d['trial_idx'], d['wav_set_idx'], d['overall_snr'], d['fg_name'], d['fg_channel'], d['bg_name'],
              d['bg_channel'], d['response_condition'], d['current_full_rep'], d['trial_is_repeat'])
        played_fg[trial_idx_py] = d['fg_name']
        played_bg[trial_idx_py] = d['bg_name']
        overall_snr[trial_idx_py] = d['overall_snr']

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
    print(Counter(overall_snr).keys())
    print(Counter(overall_snr).values())  # counts the elements' frequency

    # plot waveforms from an example trial
    # trial_idx = 0
    for trial_idx_py in np.array([0, 1]):
        trial_idx = trial_idx_py + 1
        w = fb.trial_waveform(trial_idx).T
        wb = fb.BgSet.waveform(fb.stim_list.bg_index[trial_idx])

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
# test_old_simple_category()
test_CategorySet()