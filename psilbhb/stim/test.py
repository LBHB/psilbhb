from pathlib import Path

from psilbhb.stim.bignat import BigNaturalSequenceFactory

wav_path = '/home/bburan/Documents/OHSU/LBHB/psiaudio/examples/wav-files'


def stim_queued(*args, **kwargs):
    print(f'stim queued: {args}, {kwargs}')


def stim_removed(*args, **kwargs):
    print(f'stim removed: {args}, {kwargs}')


if __name__ == "__main__":
    fs = 10e3
    n = BigNaturalSequenceFactory(fs, wav_path, duration=20, normalization='fixed',
                               norm_fixed_scale=250, fit_range=slice(6,16),
                               test_range=slice(0,2), test_reps=10)
    n.queue.connect(stim_queued, 'added')
    n.queue.connect(stim_removed, 'removed')

    # should load 4 files with ~2 sec of silence at the end of each. This
    # loads a total of 80 seconds worth of stim.
    print('Requesting first set of stim')
    w = n.next(int(800e3))

    # Now, we pause at 40 seconds. Note that the queue will report that it
    # removed all stim occuring after 40 seconds. The assumption is that the
    # code calling the `queue.pause` method has not actually played out the
    # samples after 40 seconds.
    print('Pausing')
    n.queue.pause(400e3 / fs)

    # Now, this will just generate zeros.
    print('Now requesting second set of stim')
    w = n.next(int(800e3))

    # Now, we tell the queue that we are resuming playout at 80 sec.
    print('Resuming')
    n.queue.resume(800e3 / fs)

    # We are generating 80 sec worth of stimuli. You'll see notifications that
    # stimuli have been queued at 80, 100, 120, 140 sec.
    print('Now requesting third set of stim')
    w = n.next(int(800e3))
    #import matplotlib.pyplot as plt
    #plt.figure();plt.plot(w[100000:300000])
else:
    pass
