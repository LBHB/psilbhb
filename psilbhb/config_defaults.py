'''
Default values for every psilbhb setting.

Registered with :func:`psi.config.register_defaults` when ``psilbhb`` is
imported, and through the ``psi.settings`` entry point so ``psi-config``
sees them without importing psilbhb. These resolve the same way as any psi
setting: default, then ``config.toml``, then the environment.

The Open Ephys server is not here: it belongs to psiopenephys, as
``PSIOPENEPHYS_URI``.
'''
import logging
from importlib.metadata import entry_points
from pathlib import Path

from psi import Setting, get_config, register_defaults, setting_names

log = logging.getLogger(__name__)


DEFAULTS = {
    'PSILBHB_CACHE_ROOT': Setting(
        Path, lambda: get_config('PSI_BASE_DIRECTORY') / 'cache',
        doc='joblib cache for generated stimuli.'),
    'PSILBHB_TRAINING_ROOT': Setting(
        Path, lambda: get_config('PSI_BASE_DIRECTORY') / 'training',
        doc='Where training (behavior-only) data is saved.'),
    'PSILBHB_VIDEO_ROOT': Setting(
        Path, lambda: get_config('PSI_BASE_DIRECTORY') / 'video',
        doc='Where pupil video is saved; mirrors PSI_DATA_ROOT.'),
    'PSILBHB_REMOTE_ROOT': Setting(
        Path, doc='Shared folder that behavior charts are saved under.'),
    'PSILBHB_OPENEPHYS_ROOT': Setting(
        Path, doc='Where Open Ephys records, as seen from this computer.'),
    'PSILBHB_OPENEPHYS_ROOT2': Setting(
        Path, doc='Where the second Neuropixels probe records.'),

    'PSILBHB_MYSQL_HOST': Setting(str, doc='celldb server.'),
    'PSILBHB_MYSQL_PORT': Setting(int, 3306, doc='celldb server port.'),
    'PSILBHB_MYSQL_USER': Setting(str, doc='celldb user.'),
    'PSILBHB_MYSQL_PASS': Setting(str, doc='celldb password.'),
    'PSILBHB_MYSQL_DB': Setting(str, doc='celldb database name.'),
    'PSILBHB_MYSQL_TESTMODE': Setting(
        bool, False, doc='Print celldb writes instead of making them.'),

    #: Launcher state, previously experiment_launcher.json beside psi's
    #: config.py and celldblauncher.json in the preferences folder. The
    #: launcher writes these itself.
    'PSILBHB_LAUNCHER_PLUGINS': Setting(
        dict, {},
        doc='Plugins selected by default, as {paradigm: {plugin: bool}}.'),
    'PSILBHB_LAUNCHER_EXPERIMENTER': Setting(
        str, '', doc='Experimenter last selected in the launcher.'),
    'PSILBHB_LAUNCHER_ANIMAL': Setting(
        str, '', doc='Animal last selected in the launcher.'),
    'PSILBHB_LAUNCHER_TRAINING': Setting(
        str, 'Yes', doc='Training mode last selected in the launcher.'),
    'PSILBHB_LAUNCHER_RUNCLASS': Setting(
        str, 'NTD', doc='Run class last selected in the launcher.'),
}


#: Settings saved with each recording. Every registered setting of these
#: packages is included, so a new setting is recorded without being listed
#: anywhere.
RECORDED_PREFIXES = ('PSI_', 'PSILBHB_', 'PSIOPENEPHYS_')

#: Never written to a data file: credentials, and launcher state, which
#: says nothing about the recording that the recording does not already.
SECRET = {'PSILBHB_MYSQL_USER', 'PSILBHB_MYSQL_PASS'}
NOT_RECORDED = SECRET | {name for name in DEFAULTS
                         if name.startswith('PSILBHB_LAUNCHER_')}


def recorded_settings():
    '''
    The configuration to save with a recording, as {name: str(value)}.

    Settings are registered by the package that owns them when it is
    imported. This loads the ``psi.settings`` entry points first, as
    ``psi-config`` does, so that the settings of packages this process has
    not imported (psiopenephys, in the launcher) are included too.
    '''
    for entry in entry_points(group='psi.settings'):
        try:
            register_defaults(entry.load())
        except Exception as e:
            log.warning('Could not register the settings declared by %s, so '
                        'they will not be recorded: %s', entry.name, e)
    return {name: str(get_config(name)) for name in setting_names()
            if name.startswith(RECORDED_PREFIXES) and name not in NOT_RECORDED}
