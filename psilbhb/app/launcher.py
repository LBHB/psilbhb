import logging
log = logging.getLogger(__name__)

from functools import partial
import datetime as dt
import os.path
from pathlib import Path
import subprocess

from atom.api import Atom, Bool, Enum, List, Typed, Str
import enaml
from enaml.qt.qt_application import QtApplication

with enaml.imports():
    from enaml.stdlib.message_box import critical
    from psilbhb.app.launcher_view import LauncherView

from psi import get_config
from psi.util import get_tagged_values
from psi.application import (get_default_io, list_calibrations, list_io,
                             list_preferences, load_paradigm_descriptions)

from psi.experiment.api import ParadigmDescription, paradigm_manager

# redeclare these structures here:
#from psi.application.base_launcher import SimpleLauncher, launch main_animal

class SimpleLauncher(Atom):

    io = Typed(Path)
    experiment = Typed(ParadigmDescription).tag(template=True, required=True)
    calibration = Typed(Path)
    preferences = Typed(Path)
    save_data = Bool(True)
    experimenter = Str().tag(template=True)
    note = Str().tag(template=True)

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
    available_calibrations = List()
    available_preferences = List()

    # This is a bit weird, but to set the default value to not be the first
    # item in the list, you have to call the instance with the value you want
    # to be default.
    logging_level = Enum('trace', 'debug', 'info', 'warning', 'error')('info')

    def _default_experiment(self):
        return self.experiment_choices[0]

    def _default_experiment_choices(self):
        return paradigm_manager.list_paradigms(self.experiment_type)

    def _default_available_io(self):
        return list_io()

    def _update_choices(self):
        self._update_available_calibrations()
        self._update_available_preferences()

    def _update_available_calibrations(self):
        self.available_calibrations = list_calibrations(self.io)
        if not self.available_calibrations:
            self.calibration = None
            return

        if self.calibration not in self.available_calibrations:
            for calibration in self.available_calibrations:
                if calibration.stem == 'default':
                    self.calibration = calibration
                    break
            else:
                self.calibration = self.available_calibrations[0]

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
        return get_config('DATA_ROOT')

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

    def _update(self):
        exclude = [] if self.save_data else ['experimenter', 'animal', 'ear']
        required_vals = get_tagged_values(self, 'required')
        self.can_launch = True
        for k, v in get_tagged_values(self, 'required').items():
            if k in exclude:
                continue
            if not v:
                self.can_launch = False
                return

        if self.save_data:
            log.debug(f"updating template")
            template_vals = get_tagged_values(self, 'template')
            template_vals['experiment'] = template_vals['experiment'].name
            self.base_folder = self.root_folder / self.template.format(**template_vals)
            self.wildcard = self.wildcard_template.format(**template_vals)
            log.debug(f"set basefolder={self.base_folder}")
        else:
            self.base_folder = None

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

    def launch_subprocess(self):
        args = ['psi', self.experiment.name]
        plugins = [p.name for p in self.experiment.plugins if p.selected]
        if self.save_data:
            args.append(str(self.base_folder))
        if self.preferences:
            args.extend(['--preferences', str(self.get_preferences())])
        if self.io:
            args.extend(['--io', str(self.io)])
        if self.calibration:
            args.extend(['--calibration', str(self.calibration)])
        for plugin in plugins:
            args.extend(['--plugins', plugin])
        args.extend(['--debug-level-console', self.logging_level.upper()])
        args.extend(['--debug-level-file', self.logging_level.upper()])

        log.info('Launching subprocess: %s', ' '.join(args))
        print(' '.join(args))
        subprocess.check_output(args)
        self._update_choices()

class AnimalDescription:

    def __init__(self, name, title, experiment_type, plugin_info):
        '''
        Parameters
        ----------
        name : str
            Simple name that will be used to identify experiment. Must be
            globally unique. Will often be used at the command line to start
            the experment (e.g., `psi name`).
        title : str
            Title to show in main window of the experiment as well as in any
            user experience where the user is asked to select from a list of
            experiments.
        experiment_type : {'ear', 'animal', 'cohort', 'calibration', str}
            Type of experiment. This is mainly used to organize the list of
            available experments in different user interfaces.
        plugin_info : list
            List of tuples containing information about the plugins that are
            available for this particular paradigm.
        '''
        self.name = name
        self.title = title
        self.experiment_type = experiment_type

        global paradigm_manager
        try:
            self.plugins = [PluginDescription(**d) for d in plugin_info]
            paradigm_manager.register(self)
        except Exception as exc:
            print(plugin_info)
            paradigm_manager.register(self, exc)

    def enable_plugin(self, plugin_name):
        for p in self.plugins:
            if p.name == plugin_name:
                p.selected = True
                break
        else:
            choices = ', '.join(p.name for p in self.plugins)
            raise ValueError(f'Plugin {plugin_name} not found. ' \
                             f'Valid plugins are {choices}.')

    def disable_plugin(self, plugin_name):
        for p in self.plugins:
            if p.name == plugin_name:
                p.selected = False

    def disable_all_plugins(self):
        for p in self.plugins:
            p.selected = False

class CellDbLauncher(SimpleLauncher):

    #def _default_animal(self):
    #    return self.animal_choices[0]
    #
    #def _default_animal_choices(self):
    #    return ['Prince','SlipperyJack','Test']

    animal = Str().tag(template=True, required=True)
    available_animals = ['Prince','SlipperyJack','Test']

    template = '{animal}/{experiment}/{{date_time}} {experimenter} {note} {experiment}'
    wildcard_template = '*{animal}*{experiment}'

    def _observe_animal(self, event):
        self._update()


def launch(klass, experiment_type, root_folder='DATA_ROOT', view_klass=None):
    app = QtApplication()
    load_paradigm_descriptions()
    try:
        if root_folder.endswith('_ROOT'):
            root_folder = get_config(root_folder)
        if view_klass is None:
            view_klass = LauncherView
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

