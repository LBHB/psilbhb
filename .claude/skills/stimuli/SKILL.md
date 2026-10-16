---
name: psilbhb-stimuli
description: Use when working with psilbhb stimulus generation (src/psilbhb/stim/wav_set.py) - WavSet subclasses such as FgBgSet, BigNat, VowelSet, BinauralTone, AMDetect; MCWavFileSet/load_wav; generating or plotting trial waveforms outside psiexperiment; adding a new stimulus class or parameter; or matching psi event names, trial_log.csv columns, or runclass codes (NFB, NFC, BNT, BLT, AMD, VOW, ...) to the stimuli that played.
---

# psilbhb stimuli (`src/psilbhb/stim/wav_set.py`)

## Overview

All LBHB psi stimuli are **`WavSet` subclasses**. Each one is a plain Python object that
turns a flat dict of parameters into (1) a table of every distinct stimulus
(`stim_list`, one row per `wav_set_idx`), (2) a randomized trial order
(`trial_wav_idx`), and (3) for any trial, a `(n_channels, samples)` waveform plus a
dict of trial metadata. The psiexperiment engine only *wraps* these objects
(`paradigms/behavior/wav_set_manifest.enaml`). The same classes run headless in
scripts, notebooks, and nems_db post-processing (`nems_lbhb.psi_tools.wav_set_from_parms`
rebuilds them from a saved `initial.preferences` to regenerate the stimuli).

**Keep it that way:** stimulus logic belongs in `wav_set.py` and must not import enaml
or the psi controller. The only psi dependency is `psi.get_config('PSILBHB_CACHE_ROOT')`,
plus psiaudio for calibration and resampling.

## Class → runclass → paradigm

| Class | runclass | Paradigms (`paradigms/lbhb.py`) | Sound source |
|---|---|---|---|
| `FgBgSet` | NFB | NFB, NFB-passive, NFC (+ `BigNat` continuous bg) | wav dirs `fg_path`/`bg_path`/`prb_bg_path` |
| `BigNat` | BNT | BNT; NFC continuous background (`prefix='background_'`) | wav dir `sound_path` |
| `VowelSet` | BVP / VGN / VOW (by `n_response` 0/1/≥2) | BVP, VGN, VOW | wav dir, matched by name prefix |
| `CategorySet(FgBgSet)` | CAT | CAT (own `CategorySetManifest`) | several `MCWavFileSet`s |
| `AMFusion` | AMF | AMF | synthesized AM tones |
| `AMDetect` | AMD | AMD, AMD-passive | synthesized AM tones |
| `BinauralTone` | BLT | BLT | synthesized tones |
| `BinauralToneFusion` | BTF | BTF | synthesized tones |
| `RandomTone(BinauralTone)` | FTC | FTC | synthesized tones |
| `BandpassNoise` | BNB | BNB | synthesized noise |
| `BinauralAM` | BAM | BAM | synthesized AM tones |
| `Silence` | — | continuous filler | zeros |
| `OverlappingSounds` | OLP | placeholder: `__init__` raises | — |

Per-class parameters, `stim_list` columns, and event-name formats are in
`references/classes.md`. Read it before you parse a class's trial log or edit the class.

## The WavSet contract

```
WavSet(controller=None, n_response=0, output_cal=None, n_output=2, prefix='', fs=<required>, **params)
  __init__ → update_parameters(params): setattr each param, build sub-sets (MCWavFileSet), update_calibration(), update()
  update(trial_idx)       builds self.stim_list (DataFrame) and appends one shuffled permutation of
                          range(len(stim_list)) to self.trial_wav_idx (current_full_rep += 1)
  stim_row(trial_idx | wav_set_idx)  → row of stim_list; auto-calls update() when trials run out
  trial_waveform(...)     → np.array (n_output, samples); applies calibration (see below)
  trial_parameters(...)   → dict, keys prefixed with self.prefix; always contains 'event_name'
  score_response(outcome, repeat_incorrect, trial_idx)  → may re-insert the trial (repeat)
  next(samples, channel)  continuous-output callback (concatenates successive trial waveforms)
  default_parameters      list of psi context-item dicts (class attribute)
  runclass                3-letter code (property)
```

- **`trial_idx` is 1-based** (`trial_idx=0` raises). `wav_set_idx` is a 0-based row of `stim_list`.
  Pass `wav_set_idx=` to get a stimulus with no trial-sequence side effects. This is how
  nems_db regenerates every distinct stimulus.
- Subclasses implement `update`, `_trial_waveform` and `_trial_parameters`.
  `FgBgSet`, `CategorySet`, `AMFusion`, `AMDetect` and `BandpassNoise` override `trial_waveform` directly, so they
  **skip** the base calibration step below and scale levels themselves.
- `default_parameters` entries with `'type': 'Result'` are display/log-only and are not passed to the constructor.
  Subclasses append `WavSet.default_parameters` (`primary_channel`, `random_seed`, `ramp`, `replace_background`).
- Randomization is deterministic given `random_seed` and the rep number (`np.random.RandomState`).
  The one exception is `response_condition` for ambiguous/catch trials, which draws from global `np.random`.

### Levels and calibration

- `load_wav` (lru-cached): int→float [-1,1], FFT resample to `fs`, truncate/pad to `force_duration`, normalize
  (`'pe'` peak, `'rms'` → RMS 1 with soft click compression, `'fixed'` × `norm_fixed_scale`), then
  **RMS 1.0 ≡ 80 dB SPL**, attenuated by `80 - level` dB and clipped to ±5.
- Tone classes set `tonal_stim=True`: `generate_tone(..., calibration=output_cal[ch])` sets the level itself.
- Otherwise the base `trial_waveform` scales by flat cal `get_sf(1000, 80)`, or FIR-equalizes per channel if
  `output_cal` is an `InterpCalibration` (in-ear chirp calibration).
- With `output_cal=None`, a `FlatCalibration.from_spl(80, vrms=5/√2)` is used for every channel. This is fine for offline use.
- `FgBgSet`: `fg_level`/`bg_level` are absolute dB SPL lists. A level of `0` means *that component is silent*.
  `snr = fg_level - bg_level`.

### Channels

Output arrays always have 2 channels. `primary_channel` (0/1) is the reference/target side.
In `stim_list`, channel `-1` = diotic (both, each scaled by 1/√2).
**Event-name channel tokens differ by class:** NFB and BNT write `channel+1` (so `1`/`2`, with `0` = diotic).
BLT/BTF/FTC/BAM/VOW names write the raw 0-based channel.

## Running stimuli without psi

```python
import logging; logging.disable(logging.INFO)   # wav_set logs every trial at INFO
from psilbhb.stim.wav_set import FgBgSet

p = FgBgSet.default_values()          # dict of defaults (see gotcha #1)
p.update(fg_path='/auto/data/sounds/vocalizations/v4', fg_range=[0, 1],
         bg_path='/auto/data/sounds/backgrounds/v3', bg_range=[0, 1],
         fg_level=[55], bg_level=[55], fs=44000,
         migrate_fraction=0.0, fg_switch_channels=True, reward_durations=(1.0,))
ws = FgBgSet(n_response=2, **p)       # n_response: 0 passive, 1 go/no-go, >=2 nAFC

ws.stim_list                          # every distinct stimulus (DataFrame)
w = ws.trial_waveform(trial_idx=1)    # (2, samples) at fs
d = ws.trial_parameters(trial_idx=1)  # d['event_name'] == '01Stream-0-3-1_ferretb2001R-0-3-2-0dB'
w0 = ws.trial_waveform(wav_set_idx=0) # by stimulus, no trial bookkeeping
```

Working examples: `stim/test_bnt.py`, `test_nfb.py`, `test_blt.py`, `test_bam.py`, `test_amd.py`,
`test_vgn.py`, `test_ftc.py`. They are scripts, not pytest tests, and use `matplotlib.use('QtAgg')`.
To rebuild a wav_set from a recorded psi folder, use `nems_lbhb.psi_tools.wav_set_from_parms(filepath, prefix='')`.
It maps rig sound paths (`e:/sounds`, `h:/sounds`) to `/auto/data/sounds` and re-points the index ranges
at the files listed in `trial_log.csv`.

## Gotchas (verified)

1. **`default_values()` returns the *label* for EnumParameters, not the value.** Under psi, the context
   evaluates the choice (`'No'` → `False`). Offline you get the string `'No'`, which is **truthy**:
   - `BigNat`: `include_silence='No'` adds `x_silence.wav` to the set.
   - `BinauralTone`: `switch_channels='No'`/`include_mono='No'` add the switched and mono copies (1332 rows instead of ~333).
   - `FgBgSet`: `migrate_fraction='0'` raises `TypeError` in `update()`.

   Always override every Enum parameter with its real value when you construct offline.
   `references/classes.md` lists the Enum parameters for each class.
2. `FgBgSet` with `reward_durations=()` (the default) raises `IndexError` in `trial_parameters`. Pass `(1.0,)`.
3. `FgBgSet.update()` raises `ValueError: zero-size array` when `stim_list` has more than 3 rows and every row has the
   same `fg_channel`. This happens with `fg_switch_channels=False` and no spatial-attention block.
4. **Index ranges are positions in a directory listing.** `MCWavFileSet` uses the case-insensitive sorted
   `*.wav` in `path`, and saved parameters store only indices. If a wav is added to or removed from a sound folder,
   old ranges point to different sounds (see `psi_tools.remap_ranges_by_name`). Never rename or add files in
   `/auto/data/sounds/...` folders that experiments use. Make a new versioned folder instead.
5. `VowelSet` matches names by filename prefix, and an unmatched prefix silently becomes silence.
   `AMFusion` logs `event_name='Undefined'`.
6. `MCWavFileSet` duplicates `x_silence.wav` once per fit and test block when `include_silence` is truthy.
7. `event_name` strings are a cross-repo contract. `nems_lbhb.baphy_io.psi_nfb_epoch_name` and
   `runclass.check_wav_set_against_trial_log` re-derive them and must match. If you change a name
   format in `_trial_parameters`, update nems_db too, or newly recorded data will fail to load.
8. `continuous_wav_set.py` does not import (syntax errors; work in progress). `wav_set_passive.py` is a stub.
   `wav_set_gngfix.py` and `.ipynb_checkpoints/` are stale copies, so edit only `wav_set.py`.
9. `wav_set.py` imports `psi` (`get_config`), so psiexperiment must be installed and a psi config must exist,
   even for offline use. The `psi` conda env and default `python` on the lab servers both work.

## Adding a stimulus class

1. Subclass `WavSet`. Define `default_parameters` (+ `WavSet.default_parameters.copy()`), `setdefault('group_name', ...)`,
   `runclass`, `update`, `_trial_waveform` (return `(2, samples)`), and `_trial_parameters` (include `trial_idx`,
   `wav_set_idx`, `current_full_rep`, and `event_name` or `this_name`).
2. Set `self.tonal_stim = True` in `__init__` if you calibrate per-frequency yourself.
3. Add a `ParadigmDescription` in `paradigms/lbhb.py` with `WavSetManifest` `attrs={'stim_class_name': '<Class>'}`.
4. Add a `stim/test_<runclass>.py` script that builds the class offline, and check it before you run it on a rig.
5. Tell nems_db how to parse the new runclass (`baphy_io.psi_parm_read`, `runclass.py`).
