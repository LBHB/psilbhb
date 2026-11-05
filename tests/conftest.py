import os
import tempfile

# psilbhb.stim.wav_set calls get_config('PSILBHB_CACHE_ROOT') at import time. psi
# reads settings from the environment before the config file, so point it at
# a throwaway folder. This has to happen before any test module imports
# wav_set, which is why it is done at conftest import rather than in a
# fixture.
os.environ['PSILBHB_CACHE_ROOT'] = tempfile.mkdtemp(prefix='psilbhb-test-cache-')


import numpy as np
import pytest
from scipy.io import wavfile


def pytest_addoption(parser):
    parser.addoption(
        '--update-golden', action='store_true',
        help='Rewrite the golden files (tests/golden) from the current code '
        'instead of checking against them. Review the diff before '
        'committing it.')


@pytest.fixture(scope='session', autouse=True)
def app():
    '''
    The workbench tests need the enaml application. Enaml allows only one
    per process.
    '''
    from enaml.qt.qt_application import QtApplication
    from enaml.qt.QtWidgets import QApplication
    return QApplication.instance() or QtApplication()


#: Generated sounds for NFB: as many files as its saved settings index into
#: (fg_range [28, 21], bg_range [4]).
N_FG = 30
N_BG = 6
SOUND_DURATION = 0.5
#: The same rate as the speakers (`harness.FS`), so the sounds are not resampled.
FS = 48000


@pytest.fixture(scope='session')
def sounds(tmp_path_factory):
    '''
    Folders of generated sounds (noise, seeded) for NFB.
    '''
    root = tmp_path_factory.mktemp('sounds')
    rng = np.random.default_rng(0)
    folders = {
        'fg': [f'{i:02d}_fg{i:02d}.wav' for i in range(N_FG)],
        'bg': [f'bg{i:02d}.wav' for i in range(N_BG)],
        'prb': ['00_fg00_chimera.wav', '01_other_chimera.wav'],
    }
    fs = FS
    for folder, names in folders.items():
        (root / folder).mkdir()
        for name in names:
            w = rng.normal(scale=0.1, size=int(fs * SOUND_DURATION))
            wavfile.write(root / folder / name, fs,
                          (w * 32767).astype(np.int16))
    return root
