"""
Target+background class – generic self-initiated trial



Allows yoked target and background ids maybe? Unclear how these should be controlled.

Methods:
get_background(idx) – continuous bg, temporally uncoupled from fg
get_target(idx) –  can contain background as well, waveform plus target location (or indicate no-go)
trial duration – if variable?
get_all_properties_as_dict  - for saving to log

Properties available to psi:

runclass
Target_count, background_count  - range of idx
Channel count – how many speakers (2?)
Current_target_location (for behavior assessment)

Trial duration

parameters available to user—depends on stimuli
information for aligning to backgrounds? Is this useful/necessary?



Trial structure

1. ITI.

Maintain background queue if – if there is one

Pick new bgs as needed

Get foreground info, target location

2. Nose poke -> start trial

Play foreground sound from speaker 1 and/or 2, play for Y seconds (Y<X?)

Response = lick spout 1 or 2

Timeout after Y seconds

3. Loop to 1



Natural streams – 2AFC



Backgrounds have natural bg statistics

Targets have natural fg statistics.



Tone in noise – 2AFC

No bg

Target – one or two noises with varying relationship, tone embedded in one or the other.

Or go/no-go?



Phonemes

No bg

Target one or two phone
"""

class ForegroundBackgroundEngine():
    """
    Something akin to a baphy TrialObject that manages distinct foreground
    and background (optional) factories
    """
    def __init__(self, foreground=None, background=None):
        """

        """
        self.foreground = background
        self.background = background

