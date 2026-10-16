import os
import tempfile

# psilbhb.stim.wav_set calls get_config('PSILBHB_CACHE_ROOT') at import time. psi
# reads settings from the environment before the config file, so point it at
# a throwaway folder. This has to happen before any test module imports
# wav_set, which is why it is done at conftest import rather than in a
# fixture.
os.environ['PSILBHB_CACHE_ROOT'] = tempfile.mkdtemp(prefix='psilbhb-test-cache-')
