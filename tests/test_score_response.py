'''
`WavSet.score_response`: what is recorded for each trial and which trials
are repeated, for the stimulus sets the behavior paradigms build from their
settings (as psi builds them, see `harness.load_paradigm`).

`BehaviorPlugin` calls it at the end of each trial with the score (0
invalid, 1 incorrect, 2 correct), the repeat_incorrect setting (0 never,
1 invalid only, 2 invalid and incorrect) and the trial number (from 1).
'''
import numpy as np
import pytest

from test_behavior_golden import load

INVALID, INCORRECT, CORRECT = 0, 1, 2
NEVER, INVALID_ONLY, ALWAYS = 0, 1, 2


@pytest.fixture
def make_wavset(monkeypatch, sounds):
    '''
    A new stimulus set for a paradigm's session (see
    `test_behavior_golden.SESSIONS`).
    '''
    sessions = []

    def make(paradigm, variant='saved'):
        session = load(paradigm, variant, monkeypatch, sounds)
        sessions.append(session)
        wavset = session.controller.wavset
        # The trial sequence is drawn when the first trial is prepared.
        wavset.trial_parameters(1)
        return wavset

    yield make
    for session in sessions:
        session.stop()


def is_repeated(wavset, trial):
    '''
    True if the trial after `trial` plays `trial`'s stimulus again, as a
    repeat.
    '''
    return bool(wavset.trial_is_repeat[trial]) and \
        wavset.trial_wav_idx[trial] == wavset.trial_wav_idx[trial - 1]


@pytest.mark.parametrize('mode, outcome, repeated', [
    (NEVER, INVALID, False),
    (NEVER, INCORRECT, False),
    (NEVER, CORRECT, False),
    (INVALID_ONLY, INVALID, True),
    (INVALID_ONLY, INCORRECT, False),
    (INVALID_ONLY, CORRECT, False),
    (ALWAYS, INVALID, True),
    (ALWAYS, INCORRECT, True),
    (ALWAYS, CORRECT, False),
])
@pytest.mark.parametrize('paradigm', ['AMD', 'AMF', 'NFB'])
def test_repeat_modes(paradigm, mode, outcome, repeated, make_wavset):
    wavset = make_wavset(paradigm)
    trial = 3
    # The saved settings have as few as two stimuli: draw enough trials.
    wavset.trial_parameters(trial + 1)
    before = list(wavset.trial_wav_idx)
    wavset.score_response(outcome, repeat_incorrect=mode, trial_idx=trial)
    after = list(wavset.trial_wav_idx)
    if repeated:
        # The trial's stimulus is inserted right after it; the rest of the
        # sequence moves back one.
        assert after == before[:trial] + [before[trial - 1]] + before[trial:]
        assert is_repeated(wavset, trial)
        assert wavset.trial_parameters(trial + 1)['trial_is_repeat'] == 1
    else:
        assert after == before
        assert wavset.trial_parameters(trial + 1)['trial_is_repeat'] == 0


def test_repeat_of_a_repeat(make_wavset):
    '''
    A repeat that is also missed is repeated again.
    '''
    wavset = make_wavset('AMD')
    wavset.trial_parameters(4)
    stim = wavset.trial_wav_idx[0]
    for trial in (1, 2, 3):
        wavset.score_response(INCORRECT, repeat_incorrect=ALWAYS,
                              trial_idx=trial)
    assert list(wavset.trial_wav_idx[:4]) == [stim] * 4
    assert list(wavset.trial_is_repeat[:4]) == [0, 1, 1, 1]


def test_outcomes_recorded(make_wavset):
    wavset = make_wavset('AMD')
    wavset.trial_parameters(4)
    wavset.score_response(CORRECT, repeat_incorrect=NEVER, trial_idx=1)
    wavset.score_response(INCORRECT, repeat_incorrect=NEVER, trial_idx=2)
    # A trial that was never scored is recorded as 0.
    wavset.score_response(CORRECT, repeat_incorrect=NEVER, trial_idx=4)
    assert list(wavset.trial_outcomes) == [CORRECT, INCORRECT, 0, CORRECT]


def test_trial_out_of_range(make_wavset):
    wavset = make_wavset('AMD')
    with pytest.raises(ValueError):
        wavset.score_response(CORRECT, repeat_incorrect=NEVER,
                              trial_idx=len(wavset.trial_wav_idx) + 1)


def test_sequence_extends(make_wavset):
    '''
    When the trials run out, the next pass through the stimuli is shuffled
    and added. It is added one trial early: preparing the last trial of a
    pass adds the next pass, so that trial already logs the next pass's
    `current_full_rep`.
    '''
    wavset = make_wavset('AMF', 'varied')
    n = len(wavset.trial_wav_idx)
    assert n == len(wavset.stim_list)
    assert sorted(wavset.trial_wav_idx) ==         sorted(wavset.stim_list.index.tolist())
    assert wavset.trial_parameters(n - 1)['current_full_rep'] == 1
    assert len(wavset.trial_wav_idx) == n
    assert wavset.trial_parameters(n)['current_full_rep'] == 2
    assert len(wavset.trial_wav_idx) == 2 * n
    assert wavset.trial_parameters(n + 1)['current_full_rep'] == 2
    assert sorted(wavset.trial_wav_idx[n:]) ==         sorted(wavset.stim_list.index.tolist())


def probe_trials(wavset, column, is_probe):
    '''
    The trials (from 1) whose stimulus is a probe.
    '''
    stim = wavset.stim_list.loc[wavset.trial_wav_idx, column].to_numpy()
    return [i + 1 for i, v in enumerate(stim) if is_probe(v)]


#: Stimuli that are never repeated after a response, and where each set
#: marks them.
PROBES = {
    # NFB probe trials.
    'NFB': ('fg_go', lambda v: v == -3),
    # A distractor 0.05 octave from the target.
    'AMD': ('dis_offset', lambda v: np.abs(v) == 0.05),
    'AMF': ('dis_offset', lambda v: np.abs(v) == 0.05),
}


@pytest.mark.parametrize('paradigm', sorted(PROBES))
@pytest.mark.parametrize('outcome, mode, repeated', [
    (INCORRECT, ALWAYS, False),
    (CORRECT, ALWAYS, False),
    # The exception is for responses: an invalid trial is still repeated.
    (INVALID, ALWAYS, True),
    (INVALID, INVALID_ONLY, True),
])
def test_probe_trials(paradigm, outcome, mode, repeated, make_wavset):
    wavset = make_wavset(paradigm, 'varied')
    trials = probe_trials(wavset, *PROBES[paradigm])
    assert trials, 'the session has no probe trials'
    trial = trials[0]
    wavset.score_response(outcome, repeat_incorrect=mode, trial_idx=trial)
    assert is_repeated(wavset, trial) == repeated


@pytest.mark.parametrize('paradigm', sorted(PROBES))
def test_other_trials_repeated(paradigm, make_wavset):
    '''
    The same session's other trials are repeated (so the probe test is
    not passing because nothing is).
    '''
    wavset = make_wavset(paradigm, 'varied')
    probes = probe_trials(wavset, *PROBES[paradigm])
    trial = next(t for t in range(1, len(wavset.trial_wav_idx))
                 if t not in probes)
    wavset.score_response(INCORRECT, repeat_incorrect=ALWAYS, trial_idx=trial)
    assert is_repeated(wavset, trial)
