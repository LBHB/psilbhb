from pathlib import Path

from psilbhb.stim.bignat import BigNaturalSequenceFactory

wav_path = '/home/bburan/Documents/OHSU/LBHB/psiaudio/examples/wav-files'


def event_logger(*args, **kwargs):
    print(f'{args}, {kwargs}')


if __name__ == "__main__":
    n = BigNaturalSequenceFactory(10000, wav_path, duration=20, normalization='fixed',
                               norm_fixed_scale=250, fit_range=slice(6,16),
                               test_range=slice(0,2), test_reps=10)
    n.queue.connect(event_logger)
    # should load 4 files with ~2 sec of silence at the end of each
    w = n.next(800000)
    #import matplotlib.pyplot as plt
    #plt.figure();plt.plot(w[100000:300000])
else:
    pass
