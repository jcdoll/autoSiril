# Siril Job Runner

Automated Siril image processing pipeline with JSON job file configuration.

## Features

- Job file-based configuration for reproducible processing
- Auto-detection of calibration requirements from FITS headers
- Temperature tolerance matching for darks/bias
- Multi-night light frame support
- Automatic master calibration building and caching
- Support for LRGB, RGB, SHO, HOO, LSHO, and LHOO workflows
- HDR support: automatic brightness-weighted blending of multiple exposures
- VeraLux stretch and enhancement (Silentium, Revela, Vectra, StarComposer)
- StarNet integration for star removal and recomposition

## Usage

```bash
# Validate a job file (check calibration availability)
uv run run-job jobs/M42.json --validate

# Dry run (show what would happen)
uv run run-job jobs/M42.json --dry-run

# Run full pipeline
uv run run-job jobs/M42.json

# Run with logging to file
uv run run-job jobs/M42.json --log

# Run every job in jobs/ with logging and continue after failures
uv run run-job jobs/ --log --continue-on-error

# Force reprocessing (ignore cached stacks)
uv run run-job jobs/M42.json --force

# Run specific stage
uv run run-job jobs/M42.json --stage preprocess

# View registration stats from previous run
uv run run-job jobs/M42.json --seq-stats
```

### pysiril (Siril Integration)

pysiril is required for actual processing. It is installed from the GitLab repository (see pyproject.toml). Make sure Siril is running with scripting enabled before running jobs.

### StarNet Installation

For star removal, first complete [StarNet setup](../docs/starnet.md). An enabled
StarNet step that fails now fails the job; baseline images alone are not a
successful starless run.

### Siril Catalog Installation (Recommended)

Install local catalogs for faster processing and offline capability.

In Siril GUI: Scripts > Python Scripts > Core > Siril_Catalog_Installer

1. Astrometry Catalog (~1.5GB): Select "Astrometry Catalog", click Install
2. SPCC Catalog (~5-10GB): Select "SPCC Catalog", enter latitude/longitude, select "Visible from latitude", click Install

Bend, Oregon: Latitude 44.06, Longitude -121.32

Without local SPCC catalog, each job spends ~60s querying Gaia DR3 online. With local catalog, this drops to a few seconds.

## Directory Structure

The system expects this directory layout:

```
<base_path>/
├── calibration/
│   ├── masters/
│   │   ├── biases/
│   │   │   └── bias_2024_01_15.fit
│   │   ├── darks/
│   │   │   └── dark_300s_-10C_2024_01_15.fit
│   │   └── flats/
│   │       └── flat_L_2024_01_15.fit
│   └── raw/
│       ├── biases/
│       │   └── 2024_01_15/
│       ├── darks/
│       │   └── 2024_01_15_-10C/
│       │       ├── 180/
│       │       └── 300/
│       └── flats/
│           └── 2024_01_15/
│               ├── L/
│               ├── R/
│               ├── G/
│               └── B/
├── M42/
│   └── 2024_01_15/
│       ├── L180/
│       ├── L30/
│       ├── R60/
│       ├── G60/
│       └── B60/
└── jobs/
    └── M42_Jan2024.json
```

Key conventions:
- Dates use underscores: `2024_01_15`
- Light folders: `{filter}{exposure}` (e.g., `L180`, `R60`). Light exposures must be positive whole seconds; fractional exposures are rejected before processing. Each `lights` entry can be a directory of `.fit` frames or an individual FITS file. List individual files to omit contaminated exposures without moving or deleting captures. Do not list a directory and its files together: that would include those exposures twice. Every requested path must contribute readable frames.
- Bias: No temperature (readout noise is temperature-independent)
- Darks: `{date}_{temp}/{exposure}/` (thermal noise is temperature-dependent)
- Masters are cached in `calibration/masters/`

## Job File Format

```json
{
  "name": "M42_Jan2024",
  "type": "LRGB",
  "calibration": {
    "bias": "2024_01_15",
    "darks": "2024_01_15",
    "flats": "2024_01_20"
  },
  "lights": {
    "L": ["M42/2024_01_15/L180", "M42/2024_01_20/L180"],
    "R": ["M42/2024_01_15/R60"],
    "G": ["M42/2024_01_15/G60"],
    "B": ["M42/2024_01_15/B60"]
  },
  "output": "M42/processed",
  "options": {
    "temp_tolerance": 2
  }
}
```

### HDR Workflow

For bright objects requiring multiple exposures, include folders with different exposure times:

```json
{
  "lights": {
    "L": ["M42/2024_01_15/L180", "M42/2024_01_15/L30"],
    "R": ["M42/2024_01_15/R180", "M42/2024_01_15/R30"]
  }
}
```

The system will:
1. Stack each exposure separately (`stack_L_180s.fit`, `stack_L_30s.fit`)
2. Cross-register all stacks
3. Blend using brightness-weighted HDR before composition

See `examples/example_lrgb_hdr_job.json` for a complete HDR example.

### Settings File

Shared processing defaults are checked in as `Config` in [config.py](config.py).
Change shared defaults there and put per-image overrides in the tracked job's
`options` field.

Copy [settings.template.json](../settings.template.json) to `settings.json` in
the repository root and set your data location. `settings.json` is ignored by Git
so machine-specific paths stay local. The template is only a setup example; it
is not loaded at runtime.

```json
{
  "base_path": "/path/to/your/astro/data"
}
```

Optional `options` in `settings.json` can hold intentional local overrides,
such as equipment settings. Avoid copying unchanged defaults into this file:
local values override later changes to the shared defaults. Existing local
files are not changed when the template changes.

Precedence, from lowest to highest: `Config` defaults, `settings.json` options,
job options.

### Output Management

Siril writes intermediate files and top-level final products into each job's resolved processed folder, such as `<base_path>/M31/processed_lrgb`. Successful full jobs and `--stage compose` runs automatically copy the final top-level outputs and job logs into `<base_path>/outputs/<target>_<type>_output`.

Keep `outputs/` for final images and `<job output>/stacks/stack_<filter>_<exposure>s.fit` for composition retries. After reviewing and approving the output, use the separate cleanup command below. It deletes only `process/`, `hdr/`, `stacks/hdr_register/`, `stacks/hdr/` and `stacks/work/` inside that job's output directory. It keeps inputs, combined stacks, final images and logs. It does not record approval; the person or agent running it must obtain approval first.

M42 enables `flip_calibration_vertical` because its light frames have the opposite vertical orientation to its calibration masters, as checked against detector defects. This creates flipped working calibration copies without changing the originals. Its existing -10 C dark override remains explicit: matching colder darks are unavailable, and validation warns about temperature mismatches. Groups requiring different calibration masters fail rather than silently using the first frame's selection.

To retry composition from the retained stacks, use `uv run run-job jobs/M42.json --stage compose`. Full runs rebuild a stack when its selected input paths or calibration master paths change, or a source is newer than the stack. A small `*.inputs.json` file beside each stack records that selection; stacks without a readable record are rebuilt once. Use `--force` after changing preprocessing settings, which are not part of this cache check. Compose-only reuses retained stacks and does not apply changed light-frame selection.

List every job file with its resolved processed folder, existence status, and archive destination:

```bash
uv run manage-outputs list
```

Manually repeat the archive step for an existing processed folder:

```bash
uv run manage-outputs archive jobs/M42.json
```

After output approval, remove one job's temporary files:

```bash
uv run manage-outputs clean jobs/M42.json
```

Remove temporary files for every job in `jobs/` (only after approving those outputs):

```bash
uv run manage-outputs clean --all
```

See `examples/` for more job file examples.

See [Image-quality corrections](../docs/image-quality.md) for the evidence behind the M42 and IC2118 job settings and SPCC checks.
