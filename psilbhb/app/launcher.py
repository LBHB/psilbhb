import logging
log = logging.getLogger(__name__)

from functools import partial
import datetime as dt
import os
import os.path
from pathlib import Path
import subprocess
from datetime import datetime
import json
import matplotlib.pyplot as plt

from atom.api import Atom, Bool, Enum, List, Typed, Str, Int, Value
import enaml
from enaml.qt.qt_application import QtApplication

with enaml.imports():
    from enaml.stdlib.message_box import critical
    from psilbhb.app.launcher_view import LauncherView

from psi import get_config, save_config
from psi.util import get_tagged_values
from psi.application import get_default_io, list_io, list_preferences

from psi.experiment.api import ParadigmDescription, paradigm_manager

from psilbhb.config_defaults import recorded_settings
from psilbhb.util.celldb import celldb, readpsievents
from psilbhb.util.plots import plot_behavior

plt.ion()


#: Name each setting had before psiexperiment's configuration rework, for
#: the settings that globalparams.json carried then. Each is saved under
#: both names so analyses still looking up the old key keep working.
NEW_OLD = {
    'PSI_DATA_ROOT': 'DATA_ROOT',
    'PSILBHB_OPENEPHYS_ROOT': 'OPENEPHYS_ROOT',
    'PSILBHB_CACHE_ROOT': 'CACHE_ROOT',
    'PSILBHB_TRAINING_ROOT': 'TRAINING_ROOT',
    'PSILBHB_VIDEO_ROOT': 'VIDEO_ROOT',
    'PSIOPENEPHYS_URI': 'OPENEPHYS_URI',
    'PSILBHB_MYSQL_HOST': 'MYSQL_HOST',
    'PSILBHB_MYSQL_DB': 'MYSQL_DB',
}


#: Launcher fields remembered between sessions, each as the setting
#: PSILBHB_LAUNCHER_<FIELD>.
LAUNCHER_SETTINGS = ['experimenter', 'animal', 'training', 'runclass']


def legacy_settings(settings):
    '''
    Add the old name of each setting in `settings` that had one.
    '''
    result = dict(settings)
    for new, old in NEW_OLD.items():
        if new in settings:
            result[old] = settings[new]
    # Not a rename: PSI_SETTINGS_ROOT replaced a pair of folders, which are
    # its preferences and layout subfolders.
    if 'PSI_SETTINGS_ROOT' in settings:
        root = Path(settings['PSI_SETTINGS_ROOT'])
        result['PREFERENCES_ROOT'] = str(root / 'preferences')
        result['LAYOUT_ROOT'] = str(root / 'layout')
    return result


class CellDbLauncher(Atom):

    io = Value()
    experiment = Typed(ParadigmDescription).tag(template=True, required=True)
    audio = Str().tag(template=True, required=True)
    preferences = Typed(Path)
    save_data = Bool(True)
    #experimenter = Str().tag(template=True)
    #note = Str().tag(template=True)

    experiment_type = Str()
    experiment_choices = List()

    root_folder = Typed(Path)
    base_folder = Typed(Path)
    wildcard = Str()
    template = '{{date_time}} {experimenter} {note} {experiment}'
    wildcard_template = '*{experiment}'
    use_prior_preferences = Bool(False)

    can_launch = Bool(False)

    available_io = List()
    available_preferences = List()

    db = celldb()
    animal_data = db.get_animals()
    user_data = db.get_users()

    experimenter = Str().tag(required=True)
    animal = Str().tag(template=True, required=True)
    siteid = Str().tag(template=True, required=True)
    training = Str('Yes').tag(required=True)
    runclass = Str('NTD').tag(required=True)
    runnumber = Str().tag(required=False)
    penname = Str().tag(required=False)
    note = Str().tag(required=False)
    channelcount = Str().tag(required=True)

    available_audio = ['Free-field', 'In-ear']
    available_animals = list(animal_data['animal'])
    available_experimenters = list(user_data['userid'])
    available_training = ['Yes', 'Physiology+behavior', 'Physiology+passive']

    training_folder = Typed(Path)

    template = '{animal}/{siteid}/{runname}'
    wildcard_template = '*{animal}*{experiment}'

    # This is a bit weird, but to set the default value to not be the first
    # item in the list, you have to call the instance with the value you want
    # to be default.
    logging_level = Enum('trace', 'debug', 'info', 'warning', 'error')('info')

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.load_settings()
        self._update_site()

    def _default_experiment(self):
        return self.experiment_choices[0]

    def _default_audio(self):
        return self.available_audio[0]

    def _default_experiment_choices(self):
        return paradigm_manager.list_paradigms(self.experiment_type)

    def _default_available_io(self):
        return list_io()

    def _update_choices(self):
        self._update_available_preferences()

    def _update_available_preferences(self):
        if not self.experiment:
            return
        self.available_preferences = list_preferences(self.experiment)
        if not self.available_preferences:
            self.preferences = None
            return

        if self.preferences not in self.available_preferences:
            for preferences in self.available_preferences:
                if preferences.stem == 'default':
                    self.preferences = preferences
                    break
            else:
                self.preferences = self.available_preferences[0]

    def _default_io(self):
        return get_default_io()

    def _default_root_folder(self):
        return get_config('PSI_DATA_ROOT')

    def _observe_io(self, event):
        self._update_choices()

    def _observe_save_data(self, event):
        self._update()

    def _observe_experiment(self, event):
        self._update_choices()
        self._update()

    def _observe_experimenter(self, event):
        self._update()

    def _observe_note(self, event):
        self._update()

    def get_preferences(self):
        if not self.use_prior_preferences:
            return self.preferences
        options = []
        for match in self.root_folder.glob(self.wildcard):
            if (match / 'final.preferences').exists():
                n = match.name.split(' ')[0]
                date = dt.datetime.strptime(n, '%Y%m%d-%H%M%S')
                options.append((date, match / 'final.preferences'))
            elif (match / 'initial.preferences').exists():
                n = match.name.split(' ')[0]
                date = dt.datetime.strptime(n, '%Y%m%d-%H%M%S')
                options.append((date, match / 'initial.preferences'))
        options.sort(reverse=True)
        if len(options):
            return options[0][1]
        m = f'Could not find prior preferences for {self.experiment_type}'
        raise ValueError(m)

    def _default_training_folder(self):
        return get_config('PSILBHB_TRAINING_ROOT')

    def _observe_animal(self, event):
        self._update_site()

    def _observe_training(self, event):
        self._update_site()

    def _observe_runnumber(self, event):
        self._update()

    def _observe_siteid(self, event):
        self._update()

    def load_settings(self):
        # Read everything before setting anything: each setattr fires
        # observers that call save_settings, which would otherwise write the
        # fields not loaded yet back over their saved values.
        values = {k: get_config(f'PSILBHB_LAUNCHER_{k.upper()}')
                  for k in LAUNCHER_SETTINGS}
        for k, v in values.items():
            setattr(self, k, v)

    def save_settings(self):
        # Called on every update, so only touch config.toml when something
        # actually changed.
        updates = {f'PSILBHB_LAUNCHER_{k.upper()}': getattr(self, k)
                   for k in LAUNCHER_SETTINGS}
        updates = {k: v for k, v in updates.items() if v != get_config(k)}
        if updates:
            save_config(updates)

    def _update_site(self):
        log.info('Running celldb launcher full update')
        if len(self.animal):
            is_training = (self.training == 'Yes')
            self.db.user = self.experimenter
            self.db.animal = self.animal
            self.db.training = is_training
            self.penname = self.db.get_current_penname(create_if_missing=False)
            site_data=self.db.get_current_site(create_if_missing=False)
            self.siteid = site_data['cellid']
            rawdata = self.db.get_rawdata(siteid=self.siteid)
            self.runnumber = str(len(rawdata)+1)
            lastpendata = self.db.last_pen_data()
            if len(lastpendata)>0:
                self.channelcount=str(lastpendata.loc[0, 'numchans'])
        self._update()

    def _update(self):
        r = self.experiment.name
        self.runclass  =r.split('-')[0]

        exclude = [] if self.save_data else ['note' ]
        required_vals = get_tagged_values(self, 'required')
        self.can_launch = True
        for k, v in get_tagged_values(self, 'required').items():
            if k in exclude:
                continue
            if not v:
                self.can_launch = False
                return

        if self.save_data & self.can_launch:
            is_training = (self.training == 'Yes')
            try:
                filenum = int(self.runnumber)
            except:
                filenum = 1
                self.runnumber = '1'
            if self.training == 'Physiology+passive':
                bcode = 'p'
            else:
                bcode = 'a'
            if is_training:
                year = datetime.today().strftime('%Y')
                datestr = datetime.today().strftime('%Y_%m_%d')
                self.base_folder = self.training_folder / self.animal / f'training{year}' / \
                                   f'{self.animal}_{datestr}_{self.runclass}_{filenum}'
            else:
                self.base_folder = self.root_folder / self.animal / self.penname / \
                                   f'{self.siteid}{filenum:02d}_{bcode}_{self.runclass}'
        else:
            self.base_folder = None
        self.save_settings()

    # overload from SimpleLauncher so we can save info to celldb
    def launch_subprocess(self):

        # moved this up from other psi stuff so that OE check will happen before
        # celldb is advanced.
        # TODO Move to metadata.enaml for processing by psi
        if self.training == 'Yes':
            behavior = 'active'
            ephys = False
            dataroot = get_config('PSI_DATA_ROOT')
            # dataroot = get_config('PSILBHB_TRAINING_ROOT')
        elif self.training == 'Physiology+behavior':
            behavior = 'active'
            ephys = True
            dataroot = get_config('PSI_DATA_ROOT')
        else:
            behavior = 'passive'
            ephys = True
            dataroot = get_config('PSI_DATA_ROOT')

        plugins = [p.id for p in self.experiment.plugins if p.selected]
        if ephys & ('psiopenephys' not in plugins):
            log.error('Need to turn on openephy plugin!!!')
            #plugins.append('psiopenephys')
            return


        oeroot = get_config('PSILBHB_OPENEPHYS_ROOT')
        #oeroot2 = get_config('PSILBHB_OPENEPHYS_ROOT2')

        rawdata = self.db.create_rawfile(siteid=self.siteid, runclass=self.runclass,
                               filenum=int(self.runnumber), behavior=behavior,
                                         note=self.note,
                               pupil=True, psi_format=True,
                               dataroot=dataroot, rawroot=oeroot)
        log.info('Created raw data entry')
        path = os.path.join(rawdata['resppath'], rawdata['parmbase'])
        rawpath = rawdata['respfile']
        psipath = rawpath.replace('/raw','')
        try:
            print('creating path', path)
            os.makedirs(path)
            print('creating path', psipath)
            os.makedirs(psipath)
        except OSError as error:
            print(error)

        ## Pass user settings to psi via environment variables.
        # 1. Copy the existing environment
        my_env = os.environ.copy()

        # TODO: add variables with PSILBHB_ prefix, corresponding to those
        # in metadata.enaml
        save_parms = ['experimenter', 'animal', 'training', 'runclass',
                      'siteid', 'runnumber', 'note', 'base_folder', 'io']
        for k in save_parms:
            my_env[f"PSILBHB_{k.upper()}"] = str(getattr(self, k))

        # 3. Pass the custom environment to check_output
        #output = subprocess.check_output(["echo", "$MY_VAR"], env=my_env, shell=True)

        # psi finds the paradigm by importing the module named in the dotted
        # path.
        args = ['psi', f'psilbhb.paradigms.lbhb.{self.experiment.name}']

        if self.save_data:
            args.append(str(self.base_folder))
        if self.preferences:
            args.extend(['--preferences', str(self.get_preferences())])
        if self.io:
            args.extend(['--io', str(self.io)])
        for plugin in plugins:
            args.extend(['--plugins', plugin])

        args.extend(['--debug-level-console', self.logging_level.upper()])
        args.extend(['--debug-level-file', self.logging_level.upper()])

        log.info('Launching subprocess: %s', ' '.join(args))
        print(' '.join(args))
        subprocess.check_output(args, env=my_env)

        try:
            print('returned from subprocess')
            print(f'psipath: {psipath}')
            print(rawdata)

            # save global parameters
            filename = psipath + "globalparams.json"
            print(f"params file: {filename}")

            save_parms = ['experimenter','animal','training','runclass','base_folder','io']
            d = {k: str(getattr(self, k)) for k in save_parms}
            for k in rawdata.keys():
                d[k] = str(rawdata[k])

            d.update(legacy_settings(recorded_settings()))

            with open(filename, 'w') as file:
                file.write(json.dumps(d))

            d, dataparm, dataperf = readpsievents(psipath, rawdata['runclass'])
            dataparm['audio'] = self.audio
            dataparm['io'] = self.io
            #if 'runclass' in d.keys():
            #    rawdata['runclass'] = d['runclass']
            self.db.sqlupdate('gDataRaw', rawdata['rawid'], d=d, idfield='id')
            self.db.save_data(rawdata['rawid'], dataparm, parmtype=0, keep_existing=False)
            self.db.save_data(rawdata['rawid'], dataperf, parmtype=1, keep_existing=False)

            plot_behavior(rawdata['rawid'])

        except:
            print('Results read/process error')
        self._update_choices()
        self._update_site()


def launch(klass, experiment_type, root_folder='PSI_DATA_ROOT', view_klass=None):
    app = QtApplication()
    # Importing the module registers the LBHB paradigms with paradigm_manager.
    import psilbhb.paradigms.lbhb
    try:
        if root_folder.endswith('_ROOT'):
            root_folder = get_config(root_folder)
        if view_klass is None:
            view_klass = LauncherView
        log.info(f"{root_folder} {experiment_type}")
        launcher = klass(root_folder=root_folder, experiment_type=experiment_type)
        view = view_klass(launcher=launcher)
        view.show()
        app.start()
        return True
    except Exception as e:
        mesg = f'Unable to load configuration data.\n\n{e}'
        critical(None, 'Software not configured', mesg)
        raise

main_db = partial(launch, CellDbLauncher, 'animal')

if __name__ == "__main__":
    f = main_db()

