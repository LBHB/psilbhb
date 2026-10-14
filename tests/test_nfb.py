'''
Tests for FgBgSet (runclass NFB).

The sound libraries on the rigs aren't needed: each test builds a small set of
synthetic wav files. The names matter. Probe files whose name contains a FG
name (after stripping any leading `NN_` index) must never be paired with that
FG.
'''
import numpy as np
import pytest
from scipy.io import wavfile

from psilbhb.stim.wav_set import FgBgSet


# make_filter needs fs/2 above 1.1 * 19 kHz (the rigs run at 100 kHz).
FS = 50000
DURATION = 0.2

FG_FILES = ['00_ferret.wav', '01_cat.wav']
BG_FILES = ['bgA.wav', 'bgB.wav']
# 05_ferret_chimera matches 00_ferret once the NN_ index is stripped, so it
# must never be paired with it.
PRB_FILES = ['05_ferret_chimera.wav', '03_monkey_chimera.wav']


def write_wavs(folder, names, seed):
    folder.mkdir()
    rng = np.random.default_rng(seed)
    for name in names:
        w = rng.normal(scale=0.1, size=int(FS * DURATION))
        wavfile.write(folder / name, FS, (w * 32767).astype(np.int16))
    return folder


@pytest.fixture
def sound_paths(tmp_path):
    return {
        'fg_path': str(write_wavs(tmp_path / 'fg', FG_FILES, 0)),
        'bg_path': str(write_wavs(tmp_path / 'bg', BG_FILES, 1)),
        'prb_bg_path': str(write_wavs(tmp_path / 'prb', PRB_FILES, 2)),
    }


def make_set(sound_paths, probes=False, **overrides):
    params = FgBgSet.default_values()
    params.update(
        fg_path=sound_paths['fg_path'], fg_range=[0, 1],
        bg_path=sound_paths['bg_path'], bg_range=[0, 1],
        prb_bg_path='', prb_bg_range=[],
        duration=DURATION, fg_level=[55], bg_level=[55],
        contra_n=0, ipsi_n=0, diotic_n=1, fg_switch_channels=False,
        combinations='all', migrate_fraction=0.0, reward_durations=(1.0,),
        random_seed=1,
    )
    if probes:
        params.update(prb_bg_path=sound_paths['prb_bg_path'],
                      prb_bg_range=[0, 1], fg_level=[55, 65], prb_f=0.25)
    params.update(overrides)
    return FgBgSet(n_response=2, fs=FS, **params)


def probe_rows(fb):
    return fb.stim_list.loc[fb.stim_list['fg_go'] == -3]


def regular_rows(fb):
    return fb.stim_list.loc[fb.stim_list['fg_go'] != -3]


def first_trial_where(fb, predicate):
    for trial_idx in range(1, len(fb.stim_list) + 1):
        if predicate(fb.trial_parameters(trial_idx)):
            return trial_idx
    raise AssertionError('No trial matched')


################################################################################
# Regular (no probe) trials
################################################################################
def test_runclass(sound_paths):
    assert make_set(sound_paths).runclass == 'NFB'


def test_stim_list_is_all_fg_bg_combinations(sound_paths):
    fb = make_set(sound_paths)
    pairs = set(zip(fb.stim_list['fg_index'], fb.stim_list['bg_index']))
    assert pairs == {(0, 0), (0, 1), (1, 0), (1, 1)}
    assert (fb.stim_list['fg_go'] == 1).all()


def test_shuffle_without_channel_switch(sound_paths):
    # With fg_switch_channels off, fg_channel is all 0. The shuffle loop used
    # to call .min() on an empty array here and crash.
    fb = make_set(sound_paths, fg_switch_channels=False)
    assert (fb.stim_list['fg_channel'] == 0).all()
    fb.trial_parameters(1)


def test_channel_switch_doubles_stim_list(sound_paths):
    fb = make_set(sound_paths, fg_switch_channels=True)
    assert len(fb.stim_list) == 8
    assert set(fb.stim_list['fg_channel']) == {0, 1}


def test_first_rep_plays_each_stimulus_once(sound_paths):
    fb = make_set(sound_paths, fg_switch_channels=True)
    n = len(fb.stim_list)
    played = [fb.trial_parameters(i)['wav_set_idx'] for i in range(1, n + 1)]
    assert sorted(played) == list(range(n))


def test_go_trial_response_condition_is_fg_spout(sound_paths):
    fb = make_set(sound_paths, fg_switch_channels=True)
    for trial_idx in range(1, len(fb.stim_list) + 1):
        d = fb.trial_parameters(trial_idx)
        assert d['trial_cat'] == 'normal'
        assert d['response_condition'] == d['fg_channel'] + 1


@pytest.mark.parametrize('outcome, repeat_incorrect, repeated', [
    (0, 1, True),   # invalid, repeat early only
    (1, 1, False),  # incorrect, repeat early only
    (1, 2, True),   # incorrect, always repeat
    (2, 2, False),  # correct
])
def test_score_response_repeats(sound_paths, outcome, repeat_incorrect, repeated):
    fb = make_set(sound_paths)
    first = fb.trial_parameters(1)['wav_set_idx']
    fb.score_response(outcome, repeat_incorrect=repeat_incorrect, trial_idx=1)
    second = fb.trial_parameters(2)
    assert (second['wav_set_idx'] == first) == repeated
    assert bool(second['trial_is_repeat']) == repeated


def test_trial_waveform(sound_paths):
    fb = make_set(sound_paths)
    w = fb.trial_waveform(1)
    assert w.shape == (2, int(FS * DURATION))
    assert np.isfinite(w).all()
    # Diotic BG, so both channels carry signal.
    assert (w.std(axis=1) > 0).all()


################################################################################
# Probe trials
################################################################################
def test_probe_never_paired_with_matching_fg(sound_paths):
    fb = make_set(sound_paths, probes=True)
    prb = probe_rows(fb)
    assert len(prb) > 0
    for _, row in prb.iterrows():
        fg = fb.FgSet.names[row['fg_index']]
        probe = fb.PrbBgSet.names[row['bg_index']]
        assert not (fg == '00_ferret.wav' and probe == '05_ferret_chimera.wav')


def test_probe_trials_limited_to_low_snr(sound_paths):
    fb = make_set(sound_paths, probes=True)
    prb = probe_rows(fb)
    reg = regular_rows(fb)
    assert (np.abs(prb['fg_level'] - prb['bg_level']) < 4).all()
    # Regular trials keep the high-SNR level.
    assert set(reg['fg_level'] - reg['bg_level']) == {0, 10}


@pytest.mark.parametrize('prb_f', [0.1, 0.25, 0.5])
def test_probe_fraction(sound_paths, prb_f):
    fb = make_set(sound_paths, probes=True, prb_f=prb_f)
    actual = len(probe_rows(fb)) / len(fb.stim_list)
    assert actual == pytest.approx(prb_f, abs=0.01)


def test_probe_trial_parameters(sound_paths):
    fb = make_set(sound_paths, probes=True, reward_ambiguous_frac=1.0)
    trial_idx = first_trial_where(fb, lambda d: d['trial_cat'] == 'probe')
    d = fb.trial_parameters(trial_idx)
    assert d['bg_name'] in fb.PrbBgSet.names
    assert d['response_condition'] == -1


@pytest.mark.parametrize('reward_ambiguous_frac, allowed', [
    (1.0, {-1}),    # either spout rewarded
    (0.0, {0}),     # neither spout rewarded
    (0.5, {1, 2}),  # one spout, chosen at random
])
def test_probe_reward_condition(sound_paths, reward_ambiguous_frac, allowed):
    fb = make_set(sound_paths, probes=True,
                  reward_ambiguous_frac=reward_ambiguous_frac)
    for trial_idx in range(1, len(fb.stim_list) + 1):
        d = fb.trial_parameters(trial_idx)
        if d['trial_cat'] == 'probe':
            assert d['response_condition'] in allowed


def test_probe_incorrect_not_repeated(sound_paths):
    fb = make_set(sound_paths, probes=True)
    trial_idx = first_trial_where(fb, lambda d: d['trial_cat'] == 'probe')
    fb.score_response(1, repeat_incorrect=2, trial_idx=trial_idx)
    assert not fb.trial_parameters(trial_idx + 1)['trial_is_repeat']


def test_probe_trial_waveform(sound_paths):
    fb = make_set(sound_paths, probes=True)
    trial_idx = first_trial_where(fb, lambda d: d['trial_cat'] == 'probe')
    w = fb.trial_waveform(trial_idx)
    assert w.shape == (2, int(FS * DURATION))
    assert np.isfinite(w).all()


@pytest.mark.xfail(strict=True, raises=ValueError,
                   reason='prb_f <= 0 gives a zero or negative probe repeat '
                          'count and pd.concat([]) raises')
def test_probe_fraction_zero_drops_probes(sound_paths):
    fb = make_set(sound_paths, probes=True, prb_f=0)
    assert len(probe_rows(fb)) == 0
