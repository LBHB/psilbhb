'''
Golden copies of what the behavior paradigms do now, to catch any change to
the trials an animal gets or to the data they save.

Each session loads a paradigm as psi does (`harness.load_paradigm`): its
plugins, its saved preferences, and the real `BehaviorPlugin` and stimulus
set, with a scripted animal (seeded) doing the trials. For each trial, the
golden copy has:

* what the animal did;
* every event and action, with its time;
* the row sent to the trial log (every column, with its value);
* what was done with the waveforms (when they started, whether they were
  cancelled) and the sync trigger.

The stimulus table (`stim_list`) of each session and the waveforms of a few
stimuli of each paradigm are kept too.

Each paradigm runs with its saved settings and with settings that give it
more stimuli to choose from (more frequencies, distractors and probe
trials), under each repeat mode.

NFB plays the lab's sound libraries. Here it gets folders of generated
sounds, as many as its saved settings index into.

If a change is intended, rewrite the golden files with
`pytest tests/test_behavior_golden.py --update-golden` and review the diff.
'''
import json
from pathlib import Path

import numpy as np
import pytest

import harness


GOLDEN = Path(__file__).parent / 'golden'

N_TRIALS = 60
SEED = 0

#: How often the animal takes each action (see `harness.ACTIONS`), and how
#: often it first pokes too briefly to start the trial.
ACTION_P = {
    'correct': 0.45,
    'incorrect': 0.25,
    'no_response': 0.1,
    'repoke': 0.1,
    'early_np': 0.1,
}
SHORT_POKE_P = 0.1

#: Stimuli kept per paradigm (from its `varied` session).
N_STIMULI = 3

#: Settings for each session, on top of the saved preferences. `{sounds}`
#: is the folder of generated sounds (the `sounds` fixture in conftest).
NFB_SOUNDS = {
    'fg_path': '"{sounds}/fg"',
    'bg_path': '"{sounds}/bg"',
    'prb_bg_path': '"{sounds}/prb"',
}
SESSIONS = {
    ('AMF', 'saved'): {},
    ('AMF', 'varied'): {
        'target_frequency': '[1000, 2000]',
        'distractor_offset': '[0, 0.05, 1]',
        'distractor_level': '[0, 60]',
        'switch_channels': 'Yes',
        'reward_ambiguous_frac': 'random 50%',
    },
    ('AMD', 'saved'): {},
    ('AMD', 'varied'): {
        'go_frequency': '[1000, 2000]',
        'distractor_offset': '[0, 0.05]',
        'distractor_level': '[0, 55]',
        'switch_channels': 'Yes',
        'hold_duration': '0.2',
    },
    ('NFB', 'saved'): NFB_SOUNDS,
    ('NFB', 'varied'): {
        **NFB_SOUNDS,
        'prb_bg_range': '[0, 1]',
        'prb_f': '0.25',
        # Probes more than 4 dB from the background are dropped, and
        # FgBgSet divides by zero if none are left.
        'fg_level': '[60, 65]',
    },
}

#: psilbhb's repeat_incorrect choices.
REPEAT_MODES = ['No', 'Early only', 'Yes']


def session_name(paradigm, variant, repeat_mode=None):
    name = f'{paradigm}-{variant}'
    if repeat_mode is not None:
        name += '-repeat_' + repeat_mode.lower().replace(' ', '_')
    return name


SEQUENCE_PARAMS = [
    pytest.param(p, v, m, id=session_name(p, v, m))
    for p, v in SESSIONS for m in REPEAT_MODES
]
SESSION_PARAMS = [pytest.param(p, v, id=session_name(p, v))
                  for p, v in SESSIONS]


def to_json(value, sounds):
    '''
    `value` as JSON-compatible data that compares equal after a round trip,
    with the folder of generated sounds replaced by `<sounds>`.
    '''
    if isinstance(value, dict):
        return {str(k): to_json(v, sounds) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_json(v, sounds) for v in value]
    if isinstance(value, np.ndarray):
        return to_json(value.tolist(), sounds)
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and np.isnan(value):
        return 'nan'
    if isinstance(value, str):
        return value.replace(sounds.as_posix(), '<sounds>')
    if hasattr(value, 'name') and hasattr(value, 'value'):  # Enum
        return f'{type(value).__name__}.{value.name}'
    return value


def load(paradigm, variant, monkeypatch, sounds, repeat_mode=None):
    np.random.seed(SEED)
    parameters = {k: v.format(sounds=sounds.as_posix())
                  for k, v in SESSIONS[paradigm, variant].items()}
    if repeat_mode is not None:
        parameters['repeat_incorrect'] = repeat_mode
    return harness.load_paradigm(paradigm, monkeypatch, parameters)


def run_session(session, n_trials=N_TRIALS):
    '''
    The trials of `session` with the scripted animal, and each trial's
    waveforms.
    '''
    plugin = session.controller
    animal = np.random.default_rng(SEED + 1)
    session.start()
    trials, waveforms = [], []
    for _ in range(n_trials):
        actions = harness.available_actions(plugin)
        p = np.array([ACTION_P[a] for a in actions])
        action = str(animal.choice(actions, p=p / p.sum()))
        short_poke = bool(animal.random() < SHORT_POKE_P)
        n_events = len(plugin.h_actions)
        n_results = len(plugin.h_results)
        trial_waveforms = list(plugin.waveforms)
        sync = plugin.get_output('sync_trigger')
        n_sync = len(sync.triggers)

        harness.run_trial(plugin, action, short_poke)

        results = plugin.h_results[n_results:]
        assert len(results) == 1
        trials.append({
            'action': action,
            'short_poke': short_poke,
            'events': plugin.h_actions[n_events:],
            'log': results[0],
            'waveforms': [{'start': w.actual_ts, 'cancelled': w.cancelled,
                           'ramp_time': w.ramp_time}
                          for w in trial_waveforms],
            'sync': sync.triggers[n_sync:],
        })
        waveforms.append((results[0]['wav_set_idx'],
                          np.stack([w.samples for w in trial_waveforms])))
    return trials, waveforms


def compact(trials, sounds):
    '''
    The trials as JSON, with the trial log columns that have the same value
    on every trial (mostly the settings) kept once.
    '''
    trials = to_json(trials, sounds)
    columns = sorted(trials[0]['log'])
    for t in trials:
        assert sorted(t['log']) == columns, 'trial log columns differ'
    constant = {c: trials[0]['log'][c] for c in columns
                if all(t['log'][c] == trials[0]['log'][c] for t in trials)}
    for t in trials:
        t['log'] = {c: v for c, v in t['log'].items() if c not in constant}
    return {'columns': columns, 'constant': constant, 'trials': trials}


@pytest.fixture
def update(request):
    return request.config.getoption('--update-golden')


def check_json(path, actual, update):
    if update:
        GOLDEN.mkdir(exist_ok=True)
        path.write_text(json.dumps(actual, indent=1, sort_keys=True) + '\n')
        pytest.skip(f'{path.name} rewritten')
    return json.loads(path.read_text())


@pytest.mark.parametrize('paradigm, variant, repeat_mode', SEQUENCE_PARAMS)
def test_trial_sequence(paradigm, variant, repeat_mode, monkeypatch, sounds,
                        update):
    session = load(paradigm, variant, monkeypatch, sounds, repeat_mode)
    try:
        trials, _ = run_session(session)
    finally:
        session.stop()
    actual = compact(trials, sounds)
    name = session_name(paradigm, variant, repeat_mode)
    golden = check_json(GOLDEN / f'sequence-{name}.json', actual, update)
    assert actual['columns'] == golden['columns']
    assert actual['constant'] == golden['constant']
    assert len(actual['trials']) == len(golden['trials'])
    for i, (a, e) in enumerate(zip(actual['trials'], golden['trials'])):
        assert a == e, f'trial {i + 1}'


@pytest.mark.parametrize('paradigm, variant', SESSION_PARAMS)
def test_stim_list(paradigm, variant, monkeypatch, sounds, update):
    session = load(paradigm, variant, monkeypatch, sounds)
    try:
        wavset = session.controller.wavset
        actual = to_json({
            'class': type(wavset).__name__,
            'stim_list': wavset.stim_list.to_dict('list'),
            'trial_wav_idx': wavset.trial_wav_idx,
        }, sounds)
    finally:
        session.stop()
    path = GOLDEN / f'stim_list-{session_name(paradigm, variant)}.json'
    assert actual == check_json(path, actual, update)


def pick_stimuli(waveforms):
    '''
    The first `N_STIMULI` distinct stimuli played.
    '''
    picked = {}
    for idx, samples in waveforms:
        if idx not in picked and len(picked) < N_STIMULI:
            picked[int(idx)] = samples
    return picked


@pytest.mark.parametrize('paradigm', sorted({p for p, _ in SESSIONS}))
def test_stimuli(paradigm, monkeypatch, sounds, update):
    session = load(paradigm, 'varied', monkeypatch, sounds)
    try:
        _, waveforms = run_session(session, n_trials=20)
    finally:
        session.stop()
    picked = pick_stimuli(waveforms)
    path = GOLDEN / f'stimuli-{paradigm}.npz'
    if update:
        GOLDEN.mkdir(exist_ok=True)
        np.savez_compressed(
            path, wav_set_idx=np.array(list(picked)), fs=harness.FS,
            **{f'stim_{i}': np.asarray(s, dtype='float32')
               for i, s in enumerate(picked.values())})
        pytest.skip(f'{path.name} rewritten')
    golden = np.load(path)
    assert float(golden['fs']) == harness.FS
    assert list(golden['wav_set_idx']) == list(picked)
    for i, (idx, samples) in enumerate(picked.items()):
        expected = golden[f'stim_{i}']
        actual = np.asarray(samples, dtype='float32')
        assert actual.shape == expected.shape, idx
        scale = max(np.abs(expected).max(), 1e-12)
        np.testing.assert_allclose(actual, expected, rtol=1e-5,
                                   atol=1e-6 * scale,
                                   err_msg=f'wav_set_idx {idx}')
