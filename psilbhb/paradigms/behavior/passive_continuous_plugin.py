import logging
log = logging.getLogger(__name__)

import enum

from atom.api import Bool, Dict, Int, Str, Typed
from enaml.application import timed_call
from enaml.core.api import d_
import numpy as np

from psilbhb.stim.wav_set import WavSet
from .behavior_mixins import (BaseBehaviorPlugin, TrialState)

################################################################################
# Supporting
################################################################################
class PassiveTrialState(TrialState):
    '''
    Defines the possible states that the experiment can be in. We use an Enum to
    minimize problems that arise from typos by the programmer (e.g., they may
    accidentally set the state to "waiting_for_timeout" rather than
    "waiting_for_to").
    '''
    waiting_for_resume = 'waiting for resume'
    waiting_for_trial_end = 'waiting for trial end'
    waiting_for_iti = 'waiting for intertrial interval'


class PassiveEvent(enum.Enum):
    '''
    Defines the possible events that may occur during the course of the
    experiment.
    '''
    iti_start = ('iti', 'start')
    iti_end = ('iti', 'end')
    iti_duration_elapsed = ('iti', 'elapsed')

    trial_start = ('trial', 'start')
    trial_end = ('trial', 'end')
    trial_duration_elapsed = ('trial', 'elapsed')


################################################################################
# Plugin
################################################################################
class PassivePlugin(BaseBehaviorPlugin):
    '''
    Plugin for controlling appetitive experiments that are based on a reward.
    Eventually this should become generic enough that it can be used with
    aversive experiments as well (it may already be sufficiently generic).
    '''
    #: Used by the trial sequence selector to randomly select between go/nogo.
    rng = Typed(np.random.RandomState)

    wavset = Typed(WavSet)

    def _default_rng(self):
        return np.random.RandomState()

    def _default_trial_state(self):
        return PassiveTrialState.waiting_for_resume

    def can_modify(self):
        return self.trial_state in (
            PassiveTrialState.waiting_for_resume,
            PassiveTrialState.waiting_for_iti,
        )

    def apply_changes(self):
        if self.can_modify():
            self._apply_changes()
            return True
        return False

    def prepare_trial(self):
        self.start_trial()

    def start_trial(self):
        # Figure out next trial and set up selector.
        log.info('Starting next trial')
        self.trial += 1

        # This generates the waveforms that get sent to each output. We have
        # not implemented mulitchannel outputs, so we still have to set the`
        # waveform on each output individually. This does not actually play the
        # waveform yet. It's just ready once all other conditions have been met
        # (see `start_trial` where we actually start playing the waveform).
        w = self.wavset.trial_waveform(self.trial)
        o1 = self.get_output('output_1')
        o2 = self.get_output('output_2')
        st = self.get_output('sync_trigger')
        with o1.engine.lock:
            o1.set_waveform(w[0])
            o2.set_waveform(w[1])

        trial_duration = w.shape[-1] / o1.fs

        # Now trigger any callbacks that are listening for the trial_ready
        # event.
        self.invoke_actions('trial_ready')

        with o1.engine.lock:
            ts = self.get_ts()
            o1.start_waveform(ts + 0.1, False)
            o2.start_waveform(ts + 0.1, True)
            st.trigger(ts + 0.1, 0.5)

        self.trial_info = {
            'trial_number': self.trial,
            'trial_start': ts,
            **self.wavset.trial_parameters(self.trial),
            **self.context.get_values(),
        }

        self.invoke_actions('trial_start', ts)

        # Start a timer that will be invoked once trial duration has elapsed.
        self.trial_state = PassiveTrialState.waiting_for_trial_end
        self.start_event_timer(trial_duration, PassiveEvent.trial_duration_elapsed)

    def end_trial(self):
        ts = self.get_ts()
        self.invoke_actions('trial_end', ts, kw={'result': self.trial_info.copy()})
        self.trial_state = PassiveTrialState.waiting_for_iti
        self.start_event_timer('iti_duration', PassiveEvent.iti_duration_elapsed)

        # Apply pending changes that way any parameters (such as repeat_FA or
        # go_probability) are reflected in determining the next trial type.
        if self._apply_requested:
            self._apply_changes(False)

    def advance_state(self, state, timestamp):
        log.info(f'Advancing to {state}')
        self.trial_state = getattr(NAFCTrialState, f'waiting_for_{state}')
        self.invoke_actions(f'{state}_start', timestamp)
        elapsed_event = getattr(NAFCEvent, f'{state}_duration_elapsed')
        self.start_event_timer(f'{state}_duration', elapsed_event)

    def handle_waiting_for_trial_end(self, event, timestamp):
        if event == PassiveEvent.trial_duration_elapsed:
            self.end_trial()

    def handle_waiting_for_iti(self, event, timestamp):
        if event == PassiveEvent.iti_duration_elapsed:
            self.invoke_actions(PassiveEvent.iti_end.name, timestamp)
            if self._pause_requested:
                self.pause_experiment()
                self.trial_state = PassiveTrialState.waiting_for_resume
            else:
                self.start_trial()
