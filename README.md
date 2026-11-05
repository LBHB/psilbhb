# psilbhb

LBHB's behavior and physiology paradigms for
[psiexperiment](https://github.com/psiexperiment/psiexperiment) (`psi`).

The package provides:

* the paradigms (`psilbhb.paradigms.lbhb`): NFB, NFC, NTD, CAT, VOW, AMF, AMD,
  FTC, BNT and the rest;
* the stimulus generators they use (`psilbhb.stim`);
* the rigs' IO manifests (`config/psi/io`: weasel, badger, …), plus a
  keyboard manifest for running on any computer (see
  [The keyboard manifest](#the-keyboard-manifest));
* saved preferences and window layouts for each paradigm
  (`config/psi/preferences`, `config/psi/layout`);
* the celldb launcher (`psi-celldb`) and celldb utilities (`psilbhb.util`).

## Install

psilbhb needs Python 3.11 or newer. Clone psilbhb and install it. Its
dependencies (psiexperiment, psibehavior, psiopenephys, psivideo, cftscal and
the rest) come from PyPI:

```
conda create -n psi python=3.12
conda activate psi

git clone git@github.com:LBHB/psilbhb.git
pip install -e psilbhb
```

The `-e` (editable) install runs psilbhb from the checkout. This matters
because psi is pointed at the checkout's IO manifests, preferences and
layouts (see [Configure](#configure)).

The rigs also need the NI-DAQmx driver from National Instruments. A computer
that only runs the keyboard manifest does not.

`main` is the stable branch, and current work is on `dev`.

## Configure

psi reads its settings from a single TOML file, which `psi-config` manages.
Every setting has a default, so you only set the ones that differ. To see the
file in use and the value of every setting (and where each value came from):

```
psi-config show
```

The file is `~/psi/config.toml` unless the `PSI_CONFIG_FILE` environment
variable names another one (`psi-config --help` explains how to set it). This
is useful for keeping separate configurations, e.g. one for a rig and one for
testing on a laptop.

A rig that still has a `config.py` from before psi's configuration rework can
convert it with `psi-config migrate` (try `--dry-run` first).

### Point psi at psilbhb's IO manifests, preferences and layouts

psilbhb keeps its IO manifests, preferences and layouts in the package
(`src/psilbhb/config/psi`), so point psi at that folder in your checkout. In
the commands below, replace `<psilbhb>` with the path to the psilbhb
checkout:

```
psi-config set PSI_IO_ROOT <psilbhb>\src\psilbhb\config\psi\io
psi-config set PSI_SETTINGS_ROOT <psilbhb>\src\psilbhb\config\psi
```

* `PSI_IO_ROOT` is the folder of IO manifests (one `.enaml` file per rig).
* `PSI_SETTINGS_ROOT` holds the saved settings for each paradigm, as
  `preferences/<paradigm>/` and `layout/<paradigm>/` (e.g.,
  `preferences/NFB/default.preferences`). psi loads a paradigm's
  `default.preferences` and `default.layout` when the paradigm starts. When
  you save preferences or a layout from psi, they go to these folders, which
  are in the git checkout, so they can be committed.

### Choose the IO manifest

When psi starts without `--io`, it uses the manifest in `PSI_IO_ROOT` whose
name contains `PSI_HOSTNAME`. `PSI_HOSTNAME` defaults to the computer's name.

**For experiments on a rig**, the manifest is named after the rig (e.g.,
`weasel.enaml`). If the computer's name differs from the manifest's, set it:

```
psi-config set PSI_HOSTNAME weasel
```

**For local use on any other computer**, choose the keyboard manifest the
same way:

```
psi-config set PSI_HOSTNAME keyboard
```

Either way, you can choose a manifest for a single run by passing it to psi
with `--io` (see [Run a paradigm](#run-a-paradigm)). That overrides
`PSI_HOSTNAME`.

### Where data goes

```
psi-config set PSI_DATA_ROOT D:\data
psi-config set PSILBHB_TRAINING_ROOT D:\data\training
psi-config create-folders
```

`PSI_DATA_ROOT` holds physiology data, and `PSILBHB_TRAINING_ROOT` holds
behavior-only (training) data. `create-folders` creates every folder the
settings name. Both default to folders under `PSI_BASE_DIRECTORY`
(`~/Documents/psi`).

### LBHB settings

These are psilbhb's own settings. All of them appear in `psi-config show`,
and you set them with `psi-config set` like psi's own.

| Setting | What it is |
| --- | --- |
| `PSILBHB_CACHE_ROOT` | Cache for generated stimuli. |
| `PSILBHB_TRAINING_ROOT` | Where training (behavior-only) data is saved. |
| `PSILBHB_VIDEO_ROOT` | Where pupil video is saved. |
| `PSILBHB_REMOTE_ROOT` | Shared folder that behavior charts are saved under. |
| `PSILBHB_OPENEPHYS_ROOT`, `PSILBHB_OPENEPHYS_ROOT2` | Where Open Ephys records, as seen from this computer. |
| `PSILBHB_MYSQL_HOST`, `_PORT`, `_USER`, `_PASS`, `_DB` | The celldb server. Only needed for the celldb launcher. |
| `PSILBHB_MYSQL_TESTMODE` | Print celldb writes instead of making them. |
| `PSIOPENEPHYS_URI` | The Open Ephys server (psiopenephys's setting). |

`PSILBHB_LAUNCHER_*` settings are written by the launcher to remember your
last selections. You don't need to set them.

## Run a paradigm

With a data folder (data are saved there):

```
psi psilbhb.paradigms.lbhb.AMF D:\data\test\AMF_1
```

Without one, the paradigm runs but its data are discarded when it ends:

```
psi psilbhb.paradigms.lbhb.AMF
```

To use a particular IO manifest for one run:

```
psi psilbhb.paradigms.lbhb.AMF --io <psilbhb>\src\psilbhb\config\psi\io\keyboard.enaml
```

The paradigm is named by its path in `psilbhb.paradigms.lbhb`. The names are
the run classes: `NFB`, `NFB-passive`, `NTD-gonogo-np`, `AMF`, `FTC`, and so
on. Other useful options are `--preferences <file>` to start from other saved
settings, `--no-preferences`, `--no-layout` and `--plugins <id> ...` to turn
on optional plugins such as `psiopenephys`. Run `psi --help` for the rest.

On the rigs, `psi-celldb` opens the launcher. It creates the celldb entry for
each run, names the data folder, and saves the results to celldb afterwards.
It needs the `PSILBHB_MYSQL_*` settings and a connection to celldb.

## The keyboard manifest

`config/psi/io/keyboard.enaml` runs the paradigms on a computer without a
rig. The stimuli play on the computer's sound output, and keys on the
keyboard stand in for the IR sensors.

| Key | Input | Stands in for |
| --- | --- | --- |
| Space | `np_contact` | Nose-poke |
| Left | `spout_contact_1`, `reward_contact` | Spout 1 (the go/nogo reward spout) |
| Right | `spout_contact_2` | Spout 2 |

Hold a key to break the beam, and release it to end the contact. On the
rigs, spout 1 and the go/nogo reward spout are one sensor under two names, so
Left drives both.

### How it works

The manifest declares every channel, input and output that weasel and badger
do, so the paradigms find what they expect:

* **Sound:** `speaker_1` and `speaker_2` are the left and right channels of
  Windows' default sound output (set in Windows' Sound settings), at the
  sampling rate set for it there. On Remote Desktop, this is the remote
  computer's audio.
* **IR sensors:** each key is read as a sampled signal (10 kHz), not as an
  event. It goes high while the key is held, as the rigs' IR inputs do when
  the beam is broken. The signal then goes through the same processing as on
  a rig (low-pass filter, threshold and edge detection), so the pokes and
  spout contacts, the plots and the stored `*_analog` data behave as they do
  on a rig.
* **Monitors and microphones:** `speaker_1_monitor`, `speaker_2_monitor`,
  `microphone` (`microphone_calibration`), `microphone_1` and `microphone_2`
  are digital loopbacks. They are the samples sent to the sound card, not
  recordings.
* **Sync pulse:** `sync_trigger` is generated on the same clock as the keys,
  and read back on `sync_monitor` (`sync_trigger_monitor`). This is the same
  timing a rig records.
* **Water and lights:** the water dispensers (`water_dispense`,
  `water_dispense_1`, `water_dispense_2`) and the room light
  (`room_light_toggle`) are only logged, by the logger
  `psi.controller.engines.keyboard`.

### Things to know

* Keys are read wherever the psi window has focus, except while you are
  typing in a field. If the window loses focus, every key is released.
* The speakers are calibrated at 120 dB SPL for 1 Vrms, and the microphones
  read the same way. Levels are therefore reported at their nominal values,
  but the sounds play far quieter than on a rig rather than dangerously loud.
* Paradigms that play the lab's sound libraries (NFB, NFC, NTD, CAT, BNT, …)
  need those files. Their saved preferences point to folders on the rigs, so
  copy the sounds and change the paths in the settings to run them locally.
  The paradigms with synthetic stimuli (AMF, AMD, FTC, …) run as they are.
* For timing work, use a rig. The keyboard's timing is only as good as the
  computer's sound output, and the keys are read when Qt sees them.

## Development

```
pip install -e "psilbhb[test]"
python -m pytest psilbhb/tests
```

To work on psilbhb together with one of its dependencies, clone that package
too and install it with `pip install -e` after psilbhb. It then replaces the
PyPI release.

The tests need no rig, sound card or database.

`tests/test_behavior_golden.py` checks the behavior paradigms (AMF, AMD,
NFB) against golden copies of their trial sequences, trial logs and stimuli
in `tests/golden`. If a change to what the animal gets or what is saved is
intended, rewrite them with `python -m pytest tests/test_behavior_golden.py
--update-golden` and review the diff before committing it.
