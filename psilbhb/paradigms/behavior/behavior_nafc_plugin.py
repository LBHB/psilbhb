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
class NAFCTrialScore(enum.Enum):
    '''
    Defines the different types of scores for each trial in a behavioral
    experiment
    '''
    invalid = 0
    incorrect = 1
    correct = 2


class NAFCResponse(enum.Enum):

    no_response = 'no_response'
    early_np = 'early_np'
    np = 'np'

    # Add 
    _ignore_ = 'NAFCResponse i'
    NAFCResponse = vars()
    for i in range(2):
        NAFCResponse[f'spout_{i+1}'] = f'spout_{i+1}'
        NAFCResponse[f'early_{i+1}'] = f'early_{i+1}'


class NAFCTrialState(TrialState):
    '''
    Defines the possible states that the experiment can be in. We use an Enum to
    minimize problems that arise from typos by the programmer (e.g., they may
    accidentally set the state to "waiting_for_timeout" rather than
    "waiting_for_to").
    '''
    waiting_for_resume = 'waiting for resume'
    waiting_for_trial_start = 'waiting trial start'
    waiting_for_np_start = 'waiting nose-poke start'
    waiting_for_np_duration = 'waiting nose-poke duration'
    waiting_for_hold = 'waiting for hold'
    waiting_for_response = 'waiting for response'
    waiting_for_to = 'waiting for timeout'
    waiting_for_iti = 'waiting intertrial interval'
    waiting_for_reward_end = 'waiting break spout contact'


class NAFCEvent(enum.Enum):
    '''
    Defines the possible events that may occur during the course of the
    experiment.
    '''
    hold_start = ('hold', 'start')
    hold_duration_elapsed = ('hold', 'elapse')

    response_start = ('response', 'start')
    response_end = ('response', 'end')
    response_duration_elapsed = ('response', 'elapsed')

    _ignore_ = 'NAFCEvent i'
    NAFCEvent = vars()
    for i in range(2):
        # The third value in the tuple is the spout the animal is responding to.
        # The fourth value indicates whether it was manually triggered by the user
        # via the GUI (True) or by the animal (False).
        NAFCEvent[f'response_{i+1}_start'] = ('response', 'start', i+1, False)
        NAFCEvent[f'response_{i+1}_end'] = ('response', 'end', i+1, False)
        NAFCEvent[f'digital_response_{i+1}_start'] = ('response', 'start', i+1, True)
        NAFCEvent[f'digital_response_{i+1}_end'] = ('response', 'end', i+1, True)

    # Same rules apply as above, but for the nose-poke port.
    np_start = ('np', 'start', False, 0, False)
    np_end = ('np', 'end', False, 0, False)
    digital_np_start = ('np', 'start', 0, True)
    digital_np_end = ('np', 'end', 0, True)

    np_duration_elapsed = ('np', 'elapsed')
    to_start = ('to', 'start')
    to_end = ('to', 'end')
    to_duration_elapsed = ('to', 'elapsed')

    iti_start = ('iti', 'start')
    iti_end = ('iti', 'end')
    iti_duration_elapsed = ('iti', 'elapsed')

    trial_start = ('trial', 'start')
    trial_end = ('trial', 'end')


################################################################################
# Plugin
################################################################################
class BehaviorPlugin(BaseBehaviorPlugin):
    '''
    Plugin for controlling appetitive experiments that are based on a reward.
    Eventually this should become generic enough that it can be used with
    aversive experiments as well (it may already be sufficiently generic).
    '''
    #: Used by the trial sequence selector to randomly select between go/nogo.
    rng = Typed(np.random.RandomState)

    #: True if the user is controlling when the trials begin (e.g., in
    #: training).
    manual_control = d_(Bool(), writable=False)

    #: True if we're running in random behavior mode for debugging purposes,
    #: False otherwise.
    random_behavior_mode = Bool(False)

    next_trial_state = Str()

    wavset = Typed(WavSet)

    side = Int(-1)

    #: Number of response options (e.g., 1 for go-nogo and 2 for 2AFC). Code is
    #: written such that we can eventually add additional choices.
    N_response = Int(1)

    #: Mapping of the rising/falling edges detected on digital channels to an event.
    event_map = Dict()

    #: True if nose-poke is active
    np_active = Bool(False)

    def handle_event(self, event, timestamp=None):
        if event in (NAFCEvent.np_start, NAFCEvent.digital_np_start):
            self.np_active = True
        elif event in (NAFCEvent.np_end, NAFCEvent.digital_np_end):
            self.np_active = False
        super().handle_event(event, timestamp)

    def request_trial(self):
        log.info('Requesting trial')
        self.prepare_trial(auto_start=True)

    def _default_rng(self):
        return np.random.RandomState()

    def _default_trial_state(self):
        return NAFCTrialState.waiting_for_resume

    def _default_event_map(self):
        event_map = {
            ('rising', 'np_contact'): NAFCEvent.np_start,
            ('falling', 'np_contact'): NAFCEvent.np_end,
        }
        for i in range(self.N_response):
            event_map['rising', f'spout_contact_{i+1}'] = getattr(NAFCEvent, f'response_{i+1}_start')
            event_map['falling', f'spout_contact_{i+1}'] = getattr(NAFCEvent, f'response_{i+1}_end')
        return event_map

    def can_modify(self):
        return self.trial_state in (
            NAFCTrialState.waiting_for_resume,
            NAFCTrialState.waiting_for_trial_start,
            NAFCTrialState.waiting_for_np_start,
            NAFCTrialState.waiting_for_iti,
            NAFCTrialState.waiting_for_reward_end,
        )

    def apply_changes(self):
        if self.can_modify():
            self._apply_changes()
            return True
        return False

    def prepare_trial(self, auto_start=False):
        log.info('Preparing for next trial (auto_start %r)', auto_start)
        # Figure out next trial and set up selector.
        self.trial += 1
        self.manual_control = self.context.get_value('manual_control')
        self.trial_info = {
            'response_start': np.nan,
            'response_ts': np.nan,
            'trial_number': self.trial,
        }
        self.trial_state = NAFCTrialState.waiting_for_trial_start

        # This generates the waveforms that get sent to each output. We have
        # not implemented mulitchannel outputs, so we still have to set the`
        # waveform on each output individually. This does not actually play the
        # waveform yet. It's just ready once all other conditions have been met
        # (see `start_trial` where we actually start playing the waveform).
        w = self.wavset.trial_waveform(self.trial)
        o1 = self.get_output('output_1')
        o2 = self.get_output('output_2')
        with o1.engine.lock:
            o1.set_waveform(w[0])
            o2.set_waveform(w[1])

        # All parameters in this dictionary get logged to the trial log.
        wavset_info = self.wavset.trial_parameters(self.trial)

        # Override dispense durations
        for i in range(self.N_response):
            if f'dispense_{i+1}_duration' in wavset_info:
                if wavset_info[f'dispense_{i+1}_duration'] > 0:
                    self.context.set_value(f'water_dispense_{i+1}_duration', wavset_info[f'dispense_{i+1}_duration'])
            if 'dispense_duration' in wavset_info:
                if wavset_info[f'dispense_duration'] > 0:
                    self.context.set_value(f'water_dispense_{i+1}_duration', wavset_info[f'dispense_duration'])

        self.trial_info.update(wavset_info)
        self.side = self.trial_info['response_condition']

        # Now trigger any callbacks that are listening for the trial_ready
        # event.
        log.info('Detected nose poke state is %r', self.np_active)
        self.invoke_actions('trial_ready')
        if auto_start:
            self.start_trial()
        elif self.np_active:
            # If animal is already poking and ITI is over, go ahead and start
            # trial.
            self.trial_info['np_start'] = self.get_ts()
            self.start_trial()
        else:
            self.trial_state = NAFCTrialState.waiting_for_np_start

    def handle_waiting_for_trial_start(self, event, timestamp):
        pass

    def handle_waiting_for_np_start(self, event, timestamp):
        if event.value[:2] == ('np', 'start'):
            # Animal has nose-poked in an attempt to initiate a trial.
            self.trial_state = NAFCTrialState.waiting_for_np_duration
            self.start_event_timer('np_duration', NAFCEvent.np_duration_elapsed)
            # If the animal does not maintain the nose-poke long enough,
            # this value will be deleted.
            self.trial_info['np_start'] = timestamp

    def handle_waiting_for_np_duration(self, event, timestamp):
        if event.value[:2] == ('np', 'end'):
            # Animal has withdrawn from nose-poke too early. Cancel the timer
            # so that it does not fire a 'event_np_duration_elapsed'.
            log.debug('Animal withdrew too early')
            self.stop_event_timer()
            self.trial_state = NAFCTrialState.waiting_for_np_start
            del self.trial_info['np_start']
        elif event.value[:2] == ('np', 'elapsed'):
            log.debug('Animal initiated trial')
            self.start_trial()

    def start_trial(self):
        log.info('Starting next trial')
        # This is broken into a separate method from
        # handle_waiting_for_np_duration to allow us to trigger this method
        # from a toolbar button for training purposes.
        o1 = self.get_output('output_1')
        o2 = self.get_output('output_2')
        st = self.get_output('sync_trigger')
        with o1.engine.lock:
            ts = self.get_ts()
            o1.start_waveform(ts + 0.1, False)
            o2.start_waveform(ts + 0.1, True)
            st.trigger(ts + 0.1, 0.5)

        self.invoke_actions('trial_start', ts)
        # Notify the state machine that we are now in the hold phase of trial.
        # This means that the next time any event occurs (e.g., such as one
        # detected on the digital lines), it will call
        # `handle_waiting_for_hold` with the event and timestamp the event
        # occurred at.
        self.advance_state('hold', ts)
        self.trial_info['trial_start'] = ts

    def handle_waiting_for_hold(self, event, timestamp):
        if event.value[:2] == ('np', 'end'):
            self.invoke_actions('response_end', timestamp)
            self.trial_info['response_ts'] = timestamp
            self.trial_info['response_side'] = np.nan
            self.end_trial(NAFCResponse.early_np, NAFCTrialScore.invalid)
        elif event == NAFCEvent.hold_duration_elapsed:
            log.info('Hold duration over')
            # If we are in training mode, deliver a reward preemptively
            if self.context.get_value('training_mode') and (self.side > 0):
                self.invoke_actions(f'deliver_reward_{self.side}', timestamp)
            self.advance_state('response', timestamp)
            self.trial_info['response_start'] = timestamp

    def handle_waiting_for_response(self, event, timestamp):
        # Event is a tuple of 'response', 'start', side, True/False where False
        # indicates animal initiated event and True indicates human initiated
        # event via button. See the definbition of the NAFCEvent enum.
        log.info(f'Waiting for response. Received {event} at {timestamp}')
        if (self.N_response == 1) and (event.value[:2] == ('np', 'start')):
            # This is a special-case section for scoring go-nogo, which is
            # defined when the number of response inputs are 1. A repoke into
            # the nose port or no response will be scored as a "no" response
            # (i.e., the subject did not hear the target). A response at the
            # single response input will be socred as a "yes" response.
            self.trial_info['response_ts'] = timestamp
            self.trial_info['response_side'] = 0
            score = NAFCTrialScore.correct if self.side == 0 else NAFCTrialScore.incorrect
            response = NAFCResponse.np
            self.end_trial(response, score)
        elif (self.N_response == 1) and (event.value[:2] == ('response', 'elapsed')):
            self.trial_info['response_ts'] = np.nan
            self.trial_info['response_side'] = np.nan
            score = NAFCTrialScore.correct if self.side == 0 else NAFCTrialScore.incorrect
            response = NAFCResponse.no_response
            self.end_trial(response, score)
        elif event.value[:2] == ('response', 'start'):
            side = event.value[2]
            self.invoke_actions('response_end', timestamp)
            self.trial_info['response_ts'] = timestamp
            self.trial_info['response_side'] = side
            if self.side == -1:
                score = NAFCTrialScore.correct
                if not self.context.get_value('training_mode'):
                    self.invoke_actions(f'deliver_reward_{side}', timestamp)
            elif self.trial_info['response_side'] == self.side:
                score = NAFCTrialScore.correct
                # If we are in training mode, the reward has already been
                # delivered.
                if not self.context.get_value('training_mode'):
                    self.invoke_actions(f'deliver_reward_{side}', timestamp)
            else:
                score = NAFCTrialScore.incorrect
            response = getattr(NAFCResponse, f'spout_{side}')
            self.end_trial(response, score)
        elif event == NAFCEvent.response_duration_elapsed:
            self.invoke_actions('response_end', timestamp)
            self.trial_info['response_ts'] = np.nan
            self.trial_info['response_side'] = np.nan
            self.end_trial(NAFCResponse.no_response, NAFCTrialScore.invalid)

    def end_trial(self, response, score):
        self.stop_event_timer()
        ts = self.get_ts()
        log.info(f'Ending trial with {response} scored as {score}')

        response_time = self.trial_info['response_ts']-self.trial_info['trial_start']
        self.trial_info.update({
            'response': response.value,
            'score': score.value,
            'correct': score == NAFCTrialScore.correct,
            'response_time': response_time,
        })
        self.trial_info.update(self.context.get_values())
        self.invoke_actions('trial_end', ts, kw={'result': self.trial_info.copy()})

        if score == NAFCTrialScore.incorrect:
            # Call timeout actions and then wait for animal to withdraw from spout
            self.advance_state('to', ts)
            if response.value.startswith('spout'):
                self.start_wait_for_reward_end(ts, 'to')
        elif score == NAFCTrialScore.invalid:
            # Early withdraw from nose-poke
            # want to stop sound and start TO
            o1 = self.get_output('output_1')
            o2 = self.get_output('output_2')
            with o1.engine.lock:
                ts = self.get_ts()
                o1.stop_waveform(ts + 0.1, False)
                o2.stop_waveform(ts + 0.1, True)
            self.advance_state('to', ts)
        elif (score == NAFCTrialScore.correct) and response.value.startswith('spout'):
            # If the correct response is not a nose-poke, then this means that
            # the animal will still be on the spout. Need to wait for animal to
            # withdraw before continuing with the trial.
            self.start_wait_for_reward_end(ts, 'iti')
        else:
            self.advance_state('iti', ts)

        self.wavset.score_response(score.value,
                                   repeat_incorrect=self.context.get_value('repeat_incorrect'),
                                   trial_idx=self.trial)

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

    def start_wait_for_reward_end(self, timestamp, next_state):
        self.trial_state = NAFCTrialState.waiting_for_reward_end
        self.next_trial_state = next_state

    def handle_waiting_for_reward_end(self, event, timestamp):
        if event.value[:2] == ('response', 'end'):
            self.advance_state(self.next_trial_state, timestamp)

    def handle_waiting_for_to(self, event, timestamp):
        if event == NAFCEvent.to_duration_elapsed:
            # Turn the light back on
            self.invoke_actions('to_end', timestamp)
            self.advance_state('iti', timestamp)
        elif event.value[:2] == ('response', 'start'):
            # Cancel timeout timer and wait for animal to disconnect from lick
            # spout.
            self.stop_event_timer()
            self.start_wait_for_reward_end(timestamp, 'to')

    def handle_waiting_for_iti(self, event, timestamp):
        if event.value[:2] == ('response', 'start'):
            # Animal attempted to get reward. Reset ITI interval.
            self.stop_event_timer()
            self.start_wait_for_reward_end(timestamp, 'iti')
        elif event == NAFCEvent.iti_duration_elapsed:
            self.invoke_actions(NAFCEvent.iti_end.name, timestamp)
            if self._pause_requested:
                self.pause_experiment()
                self.trial_state = NAFCTrialState.waiting_for_resume
            else:
                self.prepare_trial()

    def start_random_behavior(self):
        log.info('Starting random behavior mode')
        self.random_behavior_mode = True
        timed_call(500, self.random_behavior_cb, NAFCEvent.digital_np_start)

    def stop_random_behavior(self):
        self.random_behavior_mode = False

    def random_behavior_cb(self, event):
        if self.random_behavior_mode:
            log.info('Handling event %r', event)
            self.handle_event(event)
            ms = np.random.uniform(100, 3000)
            if event == NAFCEvent.digital_np_start:
                next_event = NAFCEvent.digital_np_end
            else:
                next_event = NAFCEvent.digital_np_start
            log.info('Starting next event for %d ms from now', ms)
            timed_call(ms, self.random_behavior_cb, next_event)
