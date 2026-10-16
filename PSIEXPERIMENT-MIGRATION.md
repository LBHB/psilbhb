# psilbhb vs. current psiexperiment — what needs fixing

Notes from an audit on 2026-10-02 against the psi checkouts in
`projects/psi/src` (psiexperiment `3d291d2`, psibehavior `db7952e` plus
uncommitted work, psiopenephys `70c73b9`).

**Which branch.** This checkout is on `passive` (last commit 2025-03-10),
but current work is on `origin/nafc-bgsounds-ramp` (2026-05-15), which
already has Brad's March port to the new trial-log/event-log system and
psibehavior's video plugin. The list below is for
`nafc-bgsounds-ramp`; `passive` has all of this plus more (it still
imports `psi.data.sinks.api.EventLog`/`TrialLog`/`SDTAnalysis` and
`psi.application.list_calibrations`, all gone). Fix on the ramp branch.

**How it was checked.** Every `from psi…/psiaudio/cftscal/psivideo/…
import` in the package was resolved against the installed packages, and
every manifest path in `paradigms/lbhb.py` was imported. With the LBHB
settings supplied, all 19 manifests import except OpenEphys (item 1).
Nothing was run against hardware, so runtime behaviour (the NI engine,
celldb, OpenEphys itself) is unchecked.

## Blocking

1. **Every paradigm fails to load.** `paradigms/openephys.enaml` reads
   `get_config('DATA_ROOT')` at import time. psiexperiment renamed it
   `PSI_DATA_ROOT` and doesn't read the old name, so this raises and is
   re-raised as `OPENEPHYS_CONFIG_ERROR`, which wrongly blames the
   `OPENEPHYS_*` keys. `OpenEphysManifest` is in `COMMON_PLUGINS`, so all
   20 paradigms are affected. `DATA_ROOT` is only used in a
   commented-out line there, so the read can just be removed.

2. **Launcher won't start.** `app/launcher.py` imports
   `load_paradigm_descriptions` from `psi.application`. It's gone along
   with the `PARADIGM_DESCRIPTIONS` setting: paradigms are now named by
   dotted path and `paradigm_manager.get_paradigm` imports the module
   itself.
   - Replace the `load_paradigm_descriptions()` call (line ~362) with
     `import psilbhb.paradigms.lbhb` so `list_paradigms` sees them.
   - Pass the subprocess `psilbhb.paradigms.lbhb.<name>` instead of the
     bare `self.experiment.name` (line ~297); otherwise `psi` in the child
     process can't find the paradigm.

3. **Launcher view won't load.** `app/launcher_view.enaml` imports
   `get_config_folder` from `psi`, which is removed (no directory is
   inferred from the config file any more). It's used to place
   `experiment_launcher.json` (lines ~24, ~44). Put it under
   `get_config('PSI_SETTINGS_ROOT')` or `PSI_BASE_DIRECTORY`, or store the
   defaults in config.toml the way noise-exp does (`c532464`, "Store
   launcher defaults in psi's configuration file").

4. **Renamed / removed settings in `app/launcher.py`.**
   - `DATA_ROOT` → `PSI_DATA_ROOT` (lines ~131, 258, 263, 267, 332, and
     the `root_folder='DATA_ROOT'` default of `launch()` at ~360).
   - `PREFERENCES_ROOT` is removed (lines ~184, 192, 334). Preferences
     now live in `get_config('PSI_SETTINGS_ROOT') / 'preferences'`.
   - `LAYOUT_ROOT` is removed (line ~334): `PSI_SETTINGS_ROOT / 'layout'`.
   - The `config_parms` list at ~332 (saved per experiment) needs the new
     names, or it will raise on the first removed one.
   - The same old names (`DATA_ROOT`, `PREFERENCES_ROOT`, `LAYOUT_ROOT`)
     are in the stub `paradigms/metadata.enaml` (`CellDbMetdata`, lines
     ~82–107). It isn't in any paradigm yet, but it will raise once added.
     It also reads `PSILBHB_iO` (line ~78), which is probably meant to be
     `PSILBHB_IO`.

5. **Python version.** psiexperiment needs Python ≥ 3.11 (`tomllib`);
   `pyproject.toml` still says `>=3.7`.

## Rig configuration

- Each rig's `config.py` has to become `config.toml`: run
  `psi-config migrate`. It renames psi's own keys (`DATA_ROOT`,
  `IO_ROOT`, `HOSTNAME`, …) and carries the LBHB keys over unchanged,
  with a note that they have no package prefix.
- LBHB's own settings have no registered defaults, so a missing key is a
  `KeyError` at import time (`CACHE_ROOT` breaks every behavior
  manifest through `stim/bignat.py`). The full set read by the package:
  `CACHE_ROOT`, `TRAINING_ROOT`, `REMOTE_ROOT`, `VIDEO_ROOT`,
  `OPENEPHYS_URI`, `OPENEPHYS_ROOT`, `OPENEPHYS_ROOT2`, `MYSQL_HOST`,
  `MYSQL_PORT`, `MYSQL_USER`, `MYSQL_PASS`, `MYSQL_DB`, `MYSQL_TESTMODE`.
- Recommended: declare them in a `psilbhb/config_defaults.py` and
  register it as a `psi.settings` entry point, as cfts does
  (`[project.entry-points."psi.settings"]`). That gives them types and
  defaults, makes `psi-config show` list them, and is the point at which
  to give them an `LBHB_` prefix if wanted. Renaming means a
  `psi.migrations` entry point so `psi-config migrate` rewrites old keys
  (see `psi/config_migrate.py`).

## Cleanup (not blocking)

- `app/launcher.py` still has calibration plumbing (`calibration`,
  `available_calibrations`, `--calibration` at ~306). `psi` dropped that
  flag (`cab7c29`), so it would make the subprocess exit if a calibration
  were ever set. Nothing sets one now; delete it.
- `paradigms/behavior/passive_continuous.enaml` still imports
  `EventLog`/`TrialLog` from `psi.data.sinks.api` (gone). No paradigm in
  `lbhb.py` uses it; port it or delete it.
- psibehavior has uncommitted work that removes `GoNogoTrialManager`
  from `psibehavior.paradigms.api`. psilbhb doesn't use it, but recheck
  once that work lands, since psilbhb loads
  `psibehavior.paradigms.trial_log.TrialLogManifest` and `video.PSIVideo`.
- Optional, to match the other repos: move to the `src/` layout, and
  consider rebuilding the launcher on `psi.launcher` (cfts `a4ec526`,
  noise-exp).

## Related

- psiopenephys `70c73b9`: `OpenEphysClient(exe_path=...)` is now honoured,
  and `connect()` raises if Open Ephys isn't running (it used to be
  silent). psilbhb only uses `client.status`, so it isn't affected.
- psiopenephys moved to the `src/` layout; editable installs on the rigs
  need reinstalling.
