# Job File System for Siril Scripts

## Problem Statement

Current calibration scripts have hardcoded paths (e.g., `bias/08-05`, `flat/08-09/L`), making them non-reusable across different imaging sessions.

## Goal

Create a job file system that:
- Specifies lights per filter (supports multiple nights)
- References calibration files by date (auto-matches exposure/temp from FITS headers)
- Builds master calibration files if needed, caches for reuse
- Supports HDR workflows with multiple exposures per filter
- Executes via Python + sirilpy for full automation

---

## Directory Structure

**Base:** `E:\Astro\RC51_ASI2600\`

```
E:\Astro\RC51_ASI2600\
├── calibration\
│   ├── masters\
│   │   ├── biases\
│   │   │   └── bias_2024_01_15.fit
│   │   ├── darks\
│   │   │   ├── dark_300s_-10C_2024_01_15.fit
│   │   │   └── dark_60s_-10C_2024_01_15.fit
│   │   └── flats\
│   │       ├── flat_L_2024_01_15.fit
│   │       └── flat_R_2024_01_15.fit
│   └── raw\
│       ├── biases\
│       │   └── 2024_01_15\
│       ├── darks\
│       │   └── 2024_01_15_-10C\
│       │       ├── 180\
│       │       ├── 300\
│       │       └── 60\
│       └── flats\
│           └── 2024_01_15\
│               ├── L\
│               ├── R\
│               ├── G\
│               └── B\
├── M42\
│   ├── 2024_01_15\
│   │   ├── L180\
│   │   ├── L30\
│   │   ├── R60\
│   │   ├── G60\
│   │   └── B60\
│   ├── 2024_01_20\
│   │   └── L180\
│   └── processed\
└── jobs\
    └── M42_Jan2024.json
```

**Naming conventions:**
- Dates use underscores: `2024_01_15`
- Light folders: `{filter}{exposure}` (e.g., `L180`, `R60`)
- Bias raw: `{date}/` (no temperature - readout noise is temperature-independent)
- Darks raw: `{date}_{temp}/{exposure}/` (thermal noise is temperature-dependent)
- Masters: `bias_{date}.fit`, `dark_{exposure}_{temp}_{date}.fit`, `flat_{filter}_{date}.fit`

---

## Job File Format

**File:** `jobs/M42_Jan2024.json`

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
    "fwhm_filter": 1.8,
    "temp_tolerance": 2
  }
}
```

**Type options:** `LRGB`, `SHO`, `HOO`

### HDR Job Example

For bright objects requiring multiple exposures:

```json
{
  "name": "M42_HDR",
  "type": "LRGB",
  "lights": {
    "L": ["M42/2024_01_15/L180", "M42/2024_01_15/L30"],
    "R": ["M42/2024_01_15/R180", "M42/2024_01_15/R30"],
    "G": ["M42/2024_01_15/G180", "M42/2024_01_15/G30"],
    "B": ["M42/2024_01_15/B180", "M42/2024_01_15/B30"]
  }
}
```

This produces separate stacks per exposure (`stack_L_180s.fit`, `stack_L_30s.fit`).
Auto-composition is skipped; blend manually in PixInsight or other HDR tools.

---

## Auto-Matching Logic

**Step 1: Scan all light frames**
- Read FITS headers from ALL light files across all filters/nights
- Extract: exposure time, sensor temperature, filter
- Build a requirements table:

| Filter | Exposure | Temp | Count |
|--------|----------|------|-------|
| L      | 180s     | -10C | 45    |
| L      | 30s      | -10C | 20    |
| R      | 60s      | -10C | 20    |

**Step 2: Validate calibration availability**
- For each unique (exposure, temp) combo → check dark exists or can be built
- For each unique temp → check bias exists or can be built
- For each unique filter → check flat exists or can be built
- Report missing calibration before proceeding

**Step 3: Build missing masters**
- For each missing master with available raw frames → stack and cache

**Step 4: Stack by exposure**
- Frames are grouped by (filter, exposure)
- Each group stacked separately: `stack_L_180s.fit`, `stack_L_30s.fit`
- Each frame matched to appropriate dark by its own exposure/temp

---

## Why Python (not .ssf)

| Requirement | Python | .ssf |
|-------------|--------|------|
| Read JSON job files | Yes | No |
| Read FITS headers | Yes | No |
| Temperature tolerance matching | Yes | No |
| Conditional logic (if master exists...) | Yes | No |
| Progress logging | Yes | No |
| Error handling | Yes | No |
| Multi-night path handling | Yes | No |

**Decision:** All files are Python. sirilpy executes the same Siril commands.

---

## Module Structure

```
siril_job_runner/
├── __init__.py
├── calibration.py    # Calibration matching and building
├── composition.py    # RGB/LRGB/SHO composition logic
├── fits_utils.py     # FITS header reading utilities
├── job_config.py     # Job file parsing
├── job_runner.py     # Main orchestration logic
├── job_schema.json   # JSON schema for validation
├── logger.py         # Progress reporting
└── preprocessing.py  # Per-channel preprocessing with exposure grouping
```

---

## CLI Usage

```bash
# Full pipeline
uv run python run_job.py jobs/M42_Jan2024.json

# Validate only (scan headers, check calibration, report issues)
uv run python run_job.py jobs/M42_Jan2024.json --validate

# Specific stages
uv run python run_job.py jobs/M42_Jan2024.json --stage calibrate
uv run python run_job.py jobs/M42_Jan2024.json --stage preprocess
uv run python run_job.py jobs/M42_Jan2024.json --stage compose

# Dry run (show what would happen)
uv run python run_job.py jobs/M42_Jan2024.json --dry-run
```

---

## Processing Pipeline

**Stage 0: Validation**
- Scan ALL light frame headers
- Build requirements table (exposure/temp/filter combos)
- Check all required masters exist or can be built
- Fail fast with clear error if calibration is missing

**Stage 1: Calibration**
- For each unique (exposure, temp) combo → build/verify dark master
- For each unique temp → build/verify bias master
- For each unique filter → build/verify flat master

**Stage 2: Preprocessing (per filter+exposure)**
1. `convert` - Load raw lights
2. `calibrate` - Apply bias, dark, flat
3. `seqsubsky` - Background extraction
4. `register` - 2-pass registration
5. `seqapplyreg` - Apply registration with FWHM filter
6. `stack` - Winsorized sigma rejection
7. Output: `stack_{filter}_{exposure}.fit`

**Stage 3: Composition**
- If multiple exposures per filter: skip (HDR mode)
- Otherwise: register stacks, linear match, compose
- For LRGB: `rgbcomp` with luminance
- For SHO/HOO: Palette mixing via pixelmath

**Stage 4: Stretching**
- Save unstretched output for manual processing
- Apply Siril built-in stretch
- Save stretched outputs

---

## Temperature Tolerance

- Default: ±2°C matching for darks and bias
- Configurable in job options:
```json
"options": {
  "temp_tolerance": 2
}
```
- Example: Light at -9°C matches dark at -10°C

---

## Future Enhancements

- Integration with VeraLux GUI tools post-processing
- Resume from failed stage
