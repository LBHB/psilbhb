from pathlib import Path

from psilbhb.stim.bignat import BigNaturalSequenceFactory

wav_path = '/auto/data/sounds/BigNat/v2'

if False:
    # code to generate silence wav placeholder
    from scipy.io import wavfile
    wav_file = wav_path + '/seq7812.wav'
    fs, w = wavfile.read(wav_file)

    out_file = wav_path + '/x_silence.wav'
    wz = w*0
    wavfile.write(out_file, fs, wz)

def stim_queued(*args, **kwargs):
    #print(f'stim queued: {args[0]["metadata"]["filename"]}, {kwargs}')
    print(f'stim queued: {args[0]["metadata"]}, {args[0]["t0"]}, {args[0]["duration"]}, {kwargs}')


def stim_removed(*args, **kwargs):
    print(f'stim removed: {args[0]["metadata"]}, {args[0]["t0"]}, {args[0]["duration"]}, {kwargs}')


if __name__ == "__main__":
    fs = 10e3
    n = BigNaturalSequenceFactory(fs, wav_path, duration=18, normalization='fixed',
                               norm_fixed_scale=25, fit_range=slice(6,26),
                               test_range=slice(0,2), test_reps=10, channel_config='2.1', random_seed=0)
    n2 = BigNaturalSequenceFactory(fs, wav_path, duration=18, normalization='fixed',
                               norm_fixed_scale=25, fit_range=slice(6,26),
                               test_range=slice(0,2), test_reps=10, channel_config='2.2', random_seed=0)
    n.queue.connect(stim_queued, 'added')
    n.queue.connect(stim_removed, 'removed')
    n2.queue.connect(stim_queued, 'added')
    n2.queue.connect(stim_removed, 'removed')

    # should load 4 files with ~2 sec of silence at the end of each. This
    # loads a total of 80 seconds worth of stim.
    print('Requesting first set of stim')
    w = n.next(int(800e3))
    w2 = n2.next(int(800e3))

    # Now, we pause at 40 seconds. Note that the queue will report that it
    # removed all stim occuring after 40 seconds. The assumption is that the
    # code calling the `queue.pause` method has not actually played out the
    # samples after 40 seconds.
    print('Pausing')
    n.queue.pause(400e3 / fs)
    n2.queue.pause(400e3 / fs)

    # Now, this will just generate zeros.
    print('Now requesting second set of stim')
    w = n.next(int(800e3))
    w2 = n2.next(int(800e3))

    # Now, we tell the queue that we are resuming playout at 80 sec.
    print('Resuming')
    n.queue.resume(800e3 / fs)
    n2.queue.resume(800e3 / fs)

    # We are generating 80 sec worth of stimuli. You'll see notifications that
    # stimuli have been queued at 80, 100, 120, 140 sec.
    print('Now requesting third set of stim')
    w = n.next(int(800e3))
    w2 = n2.next(int(800e3))
    #import matplotlib.pyplot as plt
    #plt.figure();plt.plot(w[100000:300000])
else:
    pass
