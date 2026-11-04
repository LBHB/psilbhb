'''
Tests for the keyboard IO manifest (config/psi/io/keyboard.enaml), which runs
the paradigms on a computer without a rig. No sound card is needed: samples
are generated with explicit timestamps.
'''
from pathlib import Path
import re
from types import SimpleNamespace

import enaml
import numpy as np
import pytest

from psi.controller.api import TimedTrigger
from psi.controller.input import Callback
from psi.controller.io_manager import (
    IOManager, find_channels, find_engines, find_inputs, find_outputs,
)

IO_DIR = Path(__file__).parents[1] / 'src' / 'psilbhb' / 'config' / 'psi' / 'io'


@pytest.fixture(scope='module')
def app():
    from enaml.qt.qt_application import QtApplication
    from enaml.qt.QtWidgets import QApplication
    return QApplication.instance() or QtApplication()


@pytest.fixture
def io(app):
    '''
    The manifest's engines, channels, outputs and inputs, connected as psi
    connects them.
    '''
    with enaml.imports():
        from psilbhb.config.psi.io.keyboard import IOManifest
    manifest = IOManifest()
    point = SimpleNamespace(extensions=[
        e for e in manifest.extensions if e.point == 'psi.controller.io'
    ])
    io = IOManager()
    io.engines, io.master_engine = find_engines(point)
    io.channels = find_channels(io.engines)
    io.outputs, io.supporting = find_outputs(io.channels, point)
    io.inputs = find_inputs(io.channels, point)
    io.connect_outputs()
    io.connect_inputs()
    return io


def declared_names(path):
    return set(re.findall(r"^\s+name = '(\w+)'", path.read_text(), re.M))


@pytest.mark.parametrize('rig', ['weasel', 'badger'])
def test_has_rig_names(rig):
    '''
    Every channel, input and output a rig declares is in the keyboard
    manifest, so the paradigms find what they look for. The engines are
    named differently.
    '''
    rig_names = declared_names(IO_DIR / f'{rig}.enaml') - {'NI', 'NI_mic'}
    missing = rig_names - declared_names(IO_DIR / 'keyboard.enaml')
    assert not missing


def test_sound_card_is_master(io):
    assert io.master_engine.name == 'sound'
    assert io.engines['keyboard'].clock is io.master_engine


@pytest.mark.parametrize('names, key', [
    (['np_contact'], 'Space'),
    (['spout_contact_1', 'reward_contact'], 'Left'),
    (['spout_contact_2'], 'Right'),
])
def test_keys_are_contacts(io, names, key):
    '''
    Holding a key is a contact (a rising edge on `<name>_digital`), as the
    rigs' IR inputs go high when the beam is broken.
    '''
    engine = io.engines['keyboard']
    events = {}
    for name in names:
        assert io.channels[f'hw_ai::{name}'].key == key
        events[name] = []
        digital = io.inputs[f'{name}_digital']
        callback = Callback(function=lambda d, e=events[name]: e.extend(
            d.events[['event', 'ts']].itertuples(index=False)))
        callback.set_parent(digital)
        digital.add_input(callback)
    engine.configure()

    # In time order, as the key filter sends them.
    for name in names:
        engine.set_key(name, True, ts=1.0)
    for name in names:
        engine.set_key(name, False, ts=2.0)
    for ts in np.arange(0.1, 3.05, 0.1):
        engine.generate(ts)

    for name in names:
        assert [e for e, _ in events[name]] == ['rising', 'falling']
        assert events[name][0][1] == pytest.approx(1.0, abs=0.1)
        assert events[name][1][1] == pytest.approx(2.0, abs=0.1)


def test_sync_trigger_is_recorded(io):
    '''
    A sync pulse scheduled on `sync_trigger` (as the paradigms do) is read
    back on `sync_monitor`.
    '''
    engine = io.engines['keyboard']
    trigger = TimedTrigger(name='sync_trigger')
    io.channels['hw_do::sync_trigger'].add_output(trigger)
    blocks = []
    monitor = io.inputs['sync_trigger_monitor']
    callback = Callback(function=blocks.append)
    callback.set_parent(monitor)
    monitor.add_input(callback)
    engine.configure()

    trigger.trigger(0.5, 0.1)
    engine.generate(1.0)
    data = np.concatenate(blocks, axis=-1)[0]
    fs = engine.fs
    expected = np.zeros(int(fs))
    expected[int(0.5 * fs):int(0.6 * fs)] = 1
    np.testing.assert_array_equal(data, expected)


def test_digital_outputs_are_logged(io):
    names = {'water_dispense', 'water_dispense_1', 'water_dispense_2',
             'room_light_toggle'}
    assert names <= set(io.outputs)
