# WavSet classes: parameters, stim_list, event names

Line numbers refer to `src/psilbhb/stim/wav_set.py` as of 2026-10 and drift over time. Search for the class name.
"Enums" lists the parameters whose `default_values()` result is a label string that must be
overridden offline (see SKILL.md gotcha #1). `primary_channel` (`'0'`) is the exception:
`WavSet.__init__` casts it to int.

Example names below were produced by running each class offline with defaults (Enums fixed).

---

## Building blocks

### `load_wav(fs, filename, level, calibration=None, normalization='pe', norm_fixed_scale=1, force_duration=None)`
`lru_cache`d. Returns a 1-D float array. RMS 1 ≡ 80 dB SPL, attenuated by `80-level` dB and clipped to ±5.
`calibration` is accepted but unused.

### `WavFileSet(filenames, filelabels='', normalization, level, norm_fixed_scale, force_duration, random_seed, fs, channel_count)`
- `filenames`: a list of paths, or a list of lists (one file per output channel).
- `waveform(idx)` → `(samples, n_files)`. `names` → file names joined by `+`. `max_index` = `len(filenames)`.
- `filelabels`: per-file tag. `'C'` marks catch stimuli, which FgBgSet turns into `fg_go=-1`.

### `MCWavFileSet(path, duration=-1, include_silence=True, fit_range, fit_reps=1, test_range, test_reps=0, channel_count=1, channel_offset=1, binaural_combinations='single_offset', level=65, **WavFileSet kw)`
- Pool = case-insensitive sorted `path/*.wav`. `fit_range`/`test_range` select by **position**
  (slice, list, `None` = empty, `-1` = all fit).
- `filenames` = fit files × `fit_reps` + test files × `test_reps`. `include_silence` appends
  `x_silence.wav` (it must exist in `path`) to the fit and test blocks.
- `channel_count=2` with `'single_offset'` rotates the list by `channel_offset` for channel 2. `'all'` takes every pair.
- `cat_MCWavFileSet(a, b)` concatenates two sets with matching settings.

### `basic_sounds.py`
`generate_tone(duration, frequency, level, fs, ramp_ms, calibration)` and `generate_am_tone` are joblib-cached in
`./cachedir` (relative to the CWD!). `generate_tone_stack(freq, f_offsets, phases, bandwidth, db_depth, am, duration, fs)`
builds noise bands and harmonic stacks. `temp_seed(seed)` is a context manager for reproducible noise phases.
Without a calibration, level → amplitude `10**((level-80)/20)*5`.

### `make_filter(fs, calibration, ...)`
FIR equalizer from an `InterpCalibration`. It is used by `WavSet.trial_waveform` when `output_cal` holds in-ear chirp calibrations.

---

## FgBgSet: runclass NFB (also used by NFC)

Foreground (vocalization) over background (natural noise) with spatial configurations.
2AFC: the animal licks the spout on the FG side.

Key parameters: `fg_path`, `bg_path`, `prb_bg_path` (probe BGs; `''` → none), `fg_range`, `bg_range`,
`prb_bg_range`, `normalization`, `norm_fixed_scale`, `fg_level`/`bg_level` (lists, dB SPL, `0` = off), `duration`,
`fg_delay`, `combinations` (`'all'` = every fg×bg, `'simple'` = paired and tiled), `contra_n`/`ipsi_n`/`diotic_n` (how many
copies of each BG side relative to FG), `fg_switch_channels` (also play FG on the non-primary side), `fg_choice_trials`,
`prb_f` (regular:probe ratio), `migrate_fraction`/`migrate_start`/`migrate_stop` (FG moves sides mid-trial),
`spatial_attention_block`/`_catch_ratio`/`_start_chan` (blocked spatial attention, `SA` column),
`response_window` (tuple, s re FG onset), `reward_ambiguous_frac`, `reward_durations` (per-FG tuple).
Sub-sets: `self.FgSet`, `self.BgSet`, `self.PrbBgSet` (all `MCWavFileSet`, level 65, mono).

Enums: `normalization` (label `'rms'` happens to equal its value), `combinations` (same), `migrate_fraction`,
`reward_ambiguous_frac`, `spatial_attention_locus`.

`stim_list` columns: `index, fg_index, bg_index, fg_channel, bg_channel, fg_level, bg_level, fg_go, fg_delay, migrate_trial, SA`.
- `fg_channel`/`bg_channel`: 0, 1, or `-1` (diotic: both channels, each scaled by 1/√2).
- `fg_go`: `1` normal go (response spout = `fg_channel+1`); `-1` catch (no FG, or `'C'`-labelled) → either spout;
  `-2` choice trial (two FGs, `reward_durations` per FG); `-3` probe (BG from `PrbBgSet`, never repeated).

`trial_waveform` adds FG (delayed by `fg_delay`) to BG, scaled by `10**((level-65)/20)`, with total length
`duration`. It does **not** call the base calibration path.

`trial_parameters` keys: `trial_idx, wav_set_idx, fg_i, bg_i, fg_name, bg_name, fg_duration, bg_duration, snr,
this_fg_level, this_bg_level, this_snr, fg_delay, fg_channel, bg_channel, migrate_trial, response_condition,
response_window, current_full_rep, primary_channel, trial_is_repeat, trial_cat ('normal'|'catch'|'choice'|'probe'), SA,
dispense_duration, dispense_1_duration, dispense_2_duration, event_name`.

**event_name** = `<bg>_<fg>`:
- bg = `<bgfile>-0-<bg_dur>-<bgchan+1>` or `null`
- fg = `<fgfile>-<fg_delay>-<bg_dur>-<fgchan+1>-<snr>dB` or `null`
- Negative SNR is written `n10dB`. Channel token `0` = diotic. When the BG is off, the SNR is re-referenced to `max(bg_level)`.

Example: `01Stream-0-3-1_ferretb2001R-0-3-2-0dB`.
This must stay in sync with `nems_lbhb.baphy_io.psi_nfb_epoch_name`.

## CategorySet(FgBgSet): runclass CAT

Ferret vocalization vs. speech categorization (Satya). It is built by `configure_category_stimuli` in
`wav_set_manifest.enaml` from several `MCWavFileSet`s (`FgSet, BgSet, CatchFgSet, CatchBgSet, OAnoiseSet`).
`get_stim_list()`/`get_stim_list_no_catch()` at the top of the file choose trials from hard-coded `h:/sounds/Categories/...`
or `/auto/users/satya/...` paths, using the **global** `np.random`. Trial params include `fg_name`/`bg_name`.

## BigNat: runclass BNT (and the NFC continuous background)

Replica of baphy BigNat: long natural-sound sequences, fit set plus repeated test set, optionally binaural.

Parameters: `sound_path`, `fit_range` (default `slice(6,56)`), `test_range` (`slice(0,6)`), `test_reps`,
`include_silence`, `normalization`, `norm_fixed_scale`, `level`, `duration`, `pre_silence`, `post_silence`,
`fit_binaural`/`test_binaural` (`'none'|'oneoffset'|'twooffset'`; `'diotic'` is listed but not implemented, and fit
`'oneoffset'` raises), `binaural_index_offset`, `binaural_cross` (`'none'|'-6dB'` crossfeed), `atten_set`.
Sub-set: `self.SoundSet`. `update_parameters` pre-loads every waveform.

Enums: `include_silence`, `normalization`, `fit_binaural`, `test_binaural`, `binaural_cross`.

`stim_list`: `index, s1idx, s1_channel, s2idx, s2_channel`. `-1` = silent on that side. s1 plays on `primary_channel`.

**event_name**:
- if `test_binaural == 'none'`: `s1_name` (e.g. `seq0002.wav`)
- else: `<s1>:<s1chan+1>+<s2>:<s2chan+1>`, with an empty side written `x_silence.wav`

Waveform = `pre_silence` + sound + `post_silence`.
As a continuous NFC background (`prefix='background_'`), every logged key gets the `background_` prefix
(e.g. `background_fit_binaural`), and `next()` logs a `continuous_<channel>` event for each segment.

## VowelSet: runclass BVP (passive) / VGN (go/no-go) / VOW (2AFC)

Parameters: `sound_path`, `target_set`, `non_target_set`, `catch_set` (lists of name **prefixes**, `'A+B'` = A on the
primary side and B on the other), `switch_channels`, `mono_pairs`, `duration`, `repeat_count`, `repeat_isi`, `tar_to_cat_ratio`,
`level`, `nogo_attenuation`, `response_start`/`response_end`, `pre_silence`/`post_silence`.
Enums: `switch_channels`, `mono_pairs`.

Names are matched with `filename.startswith(prefix)`, and **an unmatched prefix silently becomes silence**. In `vowels/v5`
the files are `01_AE_106.wav`, ..., so the default `["EH_106"]` matches nothing (event name `++T`).
`stim_list`: `index, stim1idx, stim2idx, s1_channel, s2_channel, stim_cat ('T'|'N'|'C')`.
`response_condition`: 2AFC T→1, N→2, C→-1. Go/no-go T→1, N→0, C random.
**event_name** = `<s1file>:<chan>+<s2file>:<chan>+<cat>` (0-based channels).

## AMFusion: runclass AMF

Target AM tone (or noise band or harmonic stack) plus distractor tone(s) at octave offsets. 2AFC on the target side.
Parameters: `target_frequency`, `target_am_rate`, `target_bandwidth`, `modulation_depth`, `target_level`, `distractor_offset`,
`distractor_am_rate`, `distractor_level`, `harmonics`, `duration`, `swap_carriers`, `switch_channels`, `reward_ambiguous_frac`, `response_*`.
`stim_list`: `index, tar_freq, tar_am, tar_depth, tar_bandwidth, tar_level, dis_offset, dis_am, dis_level, duration, tar_channel, dis_freq, go_trial`.
It has **no `this_name` or `event_name`**, so it logs `event_name='Undefined'`. Use `target_name` (`freq:level:am`) and `distractor_name` (`freq:level`).

## AMDetect: runclass AMD

Go/no-go AM detection. Go stimuli (`go_frequency`, `go_depth`) vs. no-go (`nogo_frequency`, `nogo_depth`, `nogo_atten`), with
optional distractor, `go_multiplier`, `include_mono`, `level`, `pre_silence`/`post_silence`.
Enums: `swap_carriers`, `switch_channels`, `include_mono` (its choices are already ints), `reward_ambiguous_frac`.
`stim_list` adds `tar_cat`, `dis_channel`. The `stim_list` row with `abs(dis_offset)==0.05` is never repeated.
**event_name** = `<tarfreq>:<tarlevel>:<am>:<depth>:<tarchan>+<disfreq>:<dislevel>:0:<dischan>`, e.g. `1000.0:60:20:100:0+1000.0:0:0:0`.

## BinauralTone: runclass BLT

Passive. Reference tone on `primary_channel` plus a probe tone on the other ear. Probes span the frequency grid
`reference_center·2^±probe_octaves` (`probe_count` points) × `probe_level` (dB re ref) × `probe_delay` (ms).
A `probe_level` below -60 dB (re ref) gives a ref-only stimulus (probe freq set to the ref freq, delay 0, no probe played). Before 2026-10-13 the list wrongly flagged probes with `reference_level - probe_level > 60` (e.g. -20 dB with a 50 dB ref) as ref-only, although the probe still played at the ref frequency.
Parameters: `reference_center`, `probe_octaves`, `probe_count`, `probe_level`, `reference_level` (scalar or list, tiled
into the grid), `probe_delay`, `duration`,
`pre_silence`, `post_silence`, `switch_channels`, `include_mono`. `tonal_stim=True`: the tone is calibrated per frequency by `output_cal`.
`stim_list`: `index, name, ref_frequency, ref_level, prb_frequency, prb_level, prb_delay, duration, ref_channel, prb_channel`
(plus a spurious `level_0`). Trial params include `this_ref_level` and `this_probe_level`.
**event_name** = `<ref_f>:<ref_ch>:<ref_level>+<prb_f>:<prb_ch>:<prb_level>:<prb_delay>`, e.g. `500:0:50+500:1:-20:0`
(0-based channels). If every `probe_level <= -100`, it is `<ref_f>:<ref_ch>:<ref_level>` only.
Names written before 2026-10-13 have no `:<ref_level>` token (`500:0+500:1:-20:0`, `500:0`).

## BinauralToneFusion: runclass BTF

Same naming as the old BLT format, with no `:<ref_level>` token (`<ref_f>:<ref_ch>+<prb_f>:<prb_ch>:<prb_level>:<prb_delay>`). Adds `probe_alone_octaves` (single-probe conditions) and a `ref_level` column.
`update()` `print`s every name, which is noisy.

## RandomTone(BinauralTone): runclass FTC

Single tones over `reference_center·2^±probe_octaves` (`probe_count` points) × `reference_level` (dB SPL, scalar or
list; default `[50]`), used for frequency tuning. It forces `probe_level=[-100]`, so `update()` takes BinauralTone's single-tone branch.
`stim_list` has one row per freq × level (× 2 if `switch_channels`). The level Result is `this_ref_level`.
event_name `<f>:<chan>:<ref_level>`, e.g. `500:1:50`. Before 2026-10-13 it was `<f>:<chan>`.
nems_db `runclass._parse_stim_epochs_FTC` reads only tokens 0–1, so it ignores the level.

## BandpassNoise: runclass BNB

Noise bands (`center`, `bandwidth` in octaves, `octaves`, `count`, `level`) on one channel. It overrides `trial_waveform`
(no calibration) and scales `10**((level-80)/20)`. Its linear on/off `ramp` is in ms, capped at half the waveform.
Before 2026-10-13 it read `ramp` as seconds, so the default `ramp=10` raised a broadcast `ValueError`.
It has no `event_name`; `this_name` is used.

## BinauralAM: runclass BAM

AM reference (`am_rate`, `modulation_depth` lists) plus a probe tone or noise on the other ear (`probe_*`, `bandwidth`, `harmonics`).
**event_name** = `<ref_f>:<ref_ch>:<ref_level>:<am>:<depth>+<prb_f>:<prb_ch>:<prb_level>:<prb_delay>`, e.g. `1000:0:60:20:0+1000:1:0:0`.

## Silence

`next()` returns zeros. It is used as a continuous output placeholder.

## OverlappingSounds: runclass OLP

Placeholder. `__init__` raises `NotImplementedError`. OLP data come from baphy, not psi.

---

## What ends up on disk (psi data folder)

`initial.preferences` (YAML: `context.parameters.<group_name>.<prefix+name>` → `{expression}` or `{selected}`),
`trial_log.csv` (one row per trial: all prefixed context values plus the `trial_parameters` dict), `event_log.csv`
(timestamps; `continuous_<ch>` rows carry the continuous wav_set params), and `globalparams.json` (runclass, animal, ...).
nems_db readers: `nems_lbhb.baphy_io.psi_parm_read` (events → baphy-style exptevents) and
`nems_lbhb.psi_tools.wav_set_from_parms` (rebuild the WavSet to regenerate the stimuli).
