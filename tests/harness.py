'''
Run psilbhb's behavior paradigms as psi loads them, with the animal and the
clock under the test's control.

`load_paradigm` builds a workbench with the paradigm's plugins on the
keyboard IO manifest, applies the paradigm's saved preferences
(`config/psi/preferences/<paradigm>/default.preferences`, when there is one)
and configures the stimuli, as psi does when an experiment starts. The
controller is `HarnessPlugin`: the real `BehaviorPlugin` with these
replaced:

* the clock (`get_ts`), which only moves when the test moves it;
* timers, which are fired by hand (`fire`);
* event handling, which runs inline rather than on the next pass of Qt's
  event loop;
* the outputs, which keep the waveforms they are given rather than playing
  them;
* actions, which are recorded rather than run (so the results each trial
  sends to the trial log are kept, and no reward is dispensed).

Everything else (scoring, routing, the trial log row, the stimuli and the
trial sequence) is the paradigm's own code.
'''
from pathlib import Path
from types import SimpleNamespace

from atom.api import Float, List, Value
import enaml
import numpy as np
import yaml

from psi.core.enaml.api import load_manifest
from psi.experiment.api import Preferences, paradigm_manager

import psilbhb.paradigms.lbhb  # noqa: F401 (registers the paradigms)

with enaml.imports():
    from psilbhb.paradigms.behavior.behavior_nafc_plugin import (
        BehaviorPlugin, NAFCEvent,
    )
    from psi.application.workbench import PSIWorkbench
    from psilbhb.config.psi.io import keyboard


CONFIG = Path(__file__).parents[1] / 'src' / 'psilbhb' / 'config' / 'psi'

#: Sampling rate of the speakers. The keyboard manifest uses the computer's
#: sound output; the tests fix it (at a rate the sound card engine allows)
#: so the stimuli are the same everywhere.
FS = 48000


class FakeWaveform:

    def __init__(self, samples, ramp_time):
        self.samples = samples
        self.ramp_time = ramp_time
        self.actual_ts = np.nan
        self.cancelled = False

    def start(self, ts, *args):
        self.actual_ts = ts

    def cancel(self):
        self.cancelled = True


class FakeOutput:

    def __init__(self, name):
        self.name = name
        self.engine = SimpleNamespace(lock=_NullLock())
        self.waveforms = []
        self.triggers = []

    def add_waveform(self, samples, ramp_time=None):
        waveform = FakeWaveform(samples, ramp_time)
        self.waveforms.append(waveform)
        return waveform

    def trigger(self, ts, duration):
        self.triggers.append((ts, duration))


class _NullLock:

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


class HarnessPlugin(BehaviorPlugin):

    #: Each action invoked, as (event name, timestamp).
    h_actions = Value(factory=list)

    #: What each trial sent to the trial log.
    h_results = Value(factory=list)

    h_timer = Value()
    h_ts = Float(0.0)
    h_outputs = Value(factory=dict)

    #: The fake outputs' waveforms (the plugin requires EpochWaveforms).
    waveforms = List()

    #: Response ports the animal is touching.
    h_held = Value(factory=set)

    def invoke_actions(self, event_name, timestamp=None, delayed=False,
                       cancel_existing=True, kw=None, skip_errors=False,
                       wait=True):
        self.h_actions.append((event_name, timestamp))
        if event_name == 'trial_end':
            self.h_results.append(kw['result'])

    def get_ts(self):
        return self.h_ts

    def get_output(self, name):
        if name not in self.h_outputs:
            self.h_outputs[name] = FakeOutput(name)
        return self.h_outputs[name]

    def handle_event(self, event, timestamp=None):
        if event.name in ('np_start', 'np_end'):
            self.np_active = event.name == 'np_start'
        if event.value[:2] == ('response', 'start') and len(event.value) > 2:
            self.h_held.add(event.value[2])
        elif event.value[:2] == ('response', 'end') and len(event.value) > 2:
            self.h_held.discard(event.value[2])
        if timestamp is None:
            timestamp = self.get_ts()
        self._handle_event(event, timestamp)

    def start_event_timer(self, duration, event):
        if isinstance(duration, str):
            duration = self.context.get_value(duration)
        self.h_timer = (duration, event)

    def stop_event_timer(self):
        self.h_timer = None

    # Driving the state machine.
    def wait(self, dt):
        self.h_ts += dt

    def fire(self):
        '''
        Let the running timer elapse.
        '''
        duration, event = self.h_timer
        self.h_timer = None
        self.h_ts += duration
        self.handle_event(event)

    def event(self, name, dt=0.01):
        self.h_ts += dt
        self.handle_event(self.events[name])

    @property
    def events(self):
        return NAFCEvent

    @property
    def timer_name(self):
        return None if self.h_timer is None else self.h_timer[1].name


def plugin_manifests(paradigm):
    '''
    New instances of the paradigm's plugins (those selected by default).
    `PluginDescription.manifest` caches its instance, and a manifest
    instance cannot be registered with a second workbench.
    '''
    manifests = []
    for p in paradigm.plugins:
        if p.selected or p.required:
            manifests.append(load_manifest(p._manifest)(**p._attrs))
    return manifests


def apply_context_preferences(workbench, filename):
    '''
    Apply the context part of a saved preferences file (the parameters,
    selectors and trial log columns), as `psi.load_preferences` does. The
    rest (plots, layout) needs the GUI.
    '''
    state = yaml.load(Path(filename).read_text(), Loader=yaml.Loader)
    for extension in workbench.get_extension_point(
            'psi.experiment.preferences').extensions:
        for preference in extension.get_children(Preferences):
            if preference.name == 'context':
                preference.set_preferences(workbench, state['context'])
                return
    raise ValueError('No context preferences')


class Session:
    '''
    A paradigm loaded as psi loads it (see `load_paradigm`).
    '''

    def __init__(self, workbench, manifests):
        self.workbench = workbench
        self.manifests = manifests
        self.core = workbench.get_plugin('enaml.workbench.core')
        self.context = workbench.get_plugin('psi.context')
        self.controller = workbench.get_plugin('psi.controller')

    def set_parameters(self, parameters, apply=True):
        '''
        Set context items (name to expression, or to the label of the choice
        for an EnumParameter) and apply them.
        '''
        for name, value in parameters.items():
            item = self.context.context_items[name]
            if hasattr(item, 'choices'):
                if value not in item.choices:
                    raise ValueError(f'{name}: {value!r} is not one of '
                                     f'{list(item.choices)}')
                item.selected = value
            else:
                item.expression = str(value)
        if apply:
            self.context.apply_changes()
            self.context.next_setting()

    def configure_stimuli(self):
        '''
        Build the stimulus set from the context, as psi does when the
        experiment starts. Each call starts a new set.
        '''
        for manifest in self.manifests:
            if getattr(manifest, 'output_type', None) == 'epoch' and \
                    hasattr(manifest, 'stim_class_name'):
                self.core.invoke_command(manifest.id + '.configure_stimuli')
        return self.controller.wavset

    def start(self):
        '''
        Start the first trial, as the experiment does when it starts.
        '''
        self.controller.experiment_state = 'running'
        self.controller.prepare_trial()

    def stop(self):
        self.controller.stop()


def load_paradigm(name, monkeypatch, parameters=None):
    '''
    `name` loaded with its plugins on the keyboard IO manifest, its saved
    preferences and then `parameters` applied (see `Session.set_parameters`)
    and its stimuli configured.
    '''
    # The keyboard manifest's speakers run at the sound output's rate.
    monkeypatch.setattr(keyboard, 'default_output', lambda: ('', FS))
    paradigm = paradigm_manager.get_paradigm(name)
    manifests = plugin_manifests(paradigm)
    for m in manifests:
        if hasattr(m, 'N_response') and hasattr(m, 'factory'):
            n = m.N_response
            m.factory = lambda n=n: HarnessPlugin(N_response=n)

    workbench = PSIWorkbench()
    workbench.register_core_plugins(
        'psilbhb.config.psi.io.keyboard.IOManifest', manifests)
    session = Session(workbench, manifests)
    # psi applies the saved preferences before it checks the settings, so
    # defaults that are not valid (e.g., FgBgSet's sound folders) are never
    # evaluated when there are saved preferences.
    preferences = CONFIG / 'preferences' / name / 'default.preferences'
    if preferences.exists():
        apply_context_preferences(workbench, preferences)
    session.set_parameters(parameters or {}, apply=False)
    session.core.invoke_command('psi.context.initialize')
    session.context.apply_changes()
    session.context.next_setting()
    session.configure_stimuli()
    return session


################################################################################
# The animal
################################################################################
#: What the animal can do on a trial.
#:
#: * correct, incorrect: the response that is scored correct (or incorrect)
#:   for the trial's response condition;
#: * no_response: nothing until the response window ends;
#: * repoke: pokes the nose-poke again in the response window;
#: * early_np: withdraws from the nose-poke during the hold.
ACTIONS = ['correct', 'incorrect', 'no_response', 'repoke', 'early_np']

#: Time between the animal's own events.
DT = 0.05


def available_actions(plugin):
    '''
    The actions the animal can take on the coming trial. Withdrawing during
    the hold needs a hold.
    '''
    if plugin.context.get_value('hold_duration') > DT:
        return ACTIONS
    return [a for a in ACTIONS if a != 'early_np']


def run_trial(plugin, action, short_poke=False):
    '''
    Play one trial, starting from waiting for a nose-poke, with the animal
    doing `action` (and first poking too briefly to start the trial, if
    `short_poke`). Returns when the next trial is waiting for a nose-poke.
    '''
    S = type(plugin.trial_state)
    assert plugin.trial_state == S.waiting_for_np_start, plugin.trial_state
    trial = plugin.trial
    if short_poke and plugin.context.get_value('np_duration') > DT:
        plugin.event('np_start', DT)
        plugin.event('np_end', DT)
        assert plugin.trial_state == S.waiting_for_np_start

    plugin.event('np_start', DT)
    plugin.fire()  # np_duration: the trial starts
    if action == 'early_np':
        plugin.event('np_end', DT)
    else:
        plugin.fire()  # hold
        plugin.event('np_end', DT)
        respond(plugin, action)
    finish_trial(plugin, trial)


def respond(plugin, action):
    # AMDetect's response condition is a bool (True for go).
    side = int(plugin.side)
    n = plugin.N_response
    if action == 'no_response':
        plugin.fire()
    elif action == 'repoke':
        plugin.event('np_start', DT)
    elif action == 'correct':
        if n == 1 and side == 0:
            plugin.fire()
        else:
            press(plugin, side if side > 0 else 1)
    elif action == 'incorrect':
        if n == 1:
            # Respond to a nogo; withhold from a go.
            if side == 0:
                press(plugin, 1)
            else:
                plugin.fire()
        else:
            wrong = [p for p in range(1, n + 1) if p != side]
            if side == -1:
                # Any port is correct: nothing to get wrong.
                plugin.fire()
            else:
                press(plugin, wrong[0])
    else:
        raise ValueError(action)


def press(plugin, port):
    plugin.event(f'response_{port}_start', DT)


def finish_trial(plugin, trial, max_steps=20):
    '''
    Let the trial run out (the animal leaving the spout and nose-poke, and
    the timeout and intertrial interval elapsing) until the next trial is
    waiting for a nose-poke.
    '''
    S = type(plugin.trial_state)
    for _ in range(max_steps):
        if plugin.trial > trial and \
                plugin.trial_state == S.waiting_for_np_start:
            return
        if plugin.h_held:
            port = sorted(plugin.h_held)[0]
            plugin.event(f'response_{port}_end', DT)
        elif plugin.np_active:
            plugin.event('np_end', DT)
        elif plugin.h_timer is not None:
            plugin.fire()
        else:
            raise RuntimeError(f'Trial stuck in {plugin.trial_state}')
    raise RuntimeError(f'Trial did not end ({plugin.trial_state})')
