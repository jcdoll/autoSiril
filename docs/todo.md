# Implementation Status

The Siril Job Runner is fully implemented and operational. This document tracks the original implementation checklist and notes any remaining work.

## Code Standards

- **<300 LOC per file** - enforced, large modules split into helpers
- **Clear separation of concerns** - one responsibility per module
- **Tests via pytest** - core modules tested
- **High quality, not over-engineered** - internal single-user tool
- **No unnecessary abstractions** - kept simple and direct

---

## Core Infrastructure - COMPLETE

### Logger (`siril_job_runner/logger.py`)
- [x] `JobLogger` class with elapsed time tracking
- [x] Console output with indentation levels
- [x] Log file creation at `{output}/job_log_{timestamp}.txt`
- [x] Methods: `info()`, `step()`, `substep()`, `warning()`, `error()`
- [x] Context manager for timed operations

### FITS Utilities (`siril_job_runner/fits_utils.py`)
- [x] Read single FITS header
- [x] Extract exposure time, sensor temperature, filter name
- [x] Handle common FITS keyword variations
- [x] Scan directory of FITS files
- [x] Build requirements table from list of files

### Job Schema (`siril_job_runner/job_schema.json`)
- [x] JSON schema for job files
- [x] Required fields: name, type, calibration, lights
- [x] Optional fields: output, options
- [x] Validation for type enum: LRGB, RGB, SHO, HOO, LSHO, LHOO
- [x] Validation for calibration dates format

---

## Calibration System - COMPLETE

### Calibration Module (`siril_job_runner/calibration.py`)
- [x] `CalibrationManager` class
- [x] `find_master()` - returns path or None
- [x] `can_build_master()` - checks raw frames exist
- [x] `build_master()` - stacks and caches
- [x] Temperature tolerance matching
- [x] Master path resolution

### Calibration Path Resolution (`siril_job_runner/calibration_paths.py`)
- [x] Master and raw path conventions
- [x] Temperature-independent bias handling
- [x] Temperature-dependent dark handling

---

## Preprocessing - COMPLETE

### Preprocessing Module (`siril_job_runner/preprocessing.py`)
- [x] Process frames grouped by filter+exposure
- [x] Steps: convert, calibrate, seqsubsky, register, seqapplyreg, stack
- [x] Multi-directory light sources (multi-night)
- [x] Cached stack reuse (skip if already exists)

### Preprocessing Pipeline (`siril_job_runner/preprocessing_pipeline.py`)
- [x] Adaptive FWHM threshold filtering (GMM + dip test)
- [x] Background extraction (pre-stack)
- [x] 2-pass registration

---

## Composition - COMPLETE

### Composition Module (`siril_job_runner/composition.py`)
- [x] `Composer` class
- [x] `compose_lrgb()` - broadband with luminance
- [x] `compose_rgb()` - broadband without luminance
- [x] `compose_narrowband()` - SHO/HOO/LSHO/LHOO

### Broadband Composition (`siril_job_runner/compose_broadband.py`)
- [x] Cross-register stacks
- [x] Post-stack background extraction
- [x] SPCC color calibration
- [x] StarNet star removal
- [x] Autostretch and VeraLux stretch

### Narrowband Composition (`siril_job_runner/compose_narrowband.py`)
- [x] Channel balancing (linear_match to H)
- [x] Palette formula application
- [x] Per-channel star removal
- [x] Star compositing

### HDR Blending (`siril_job_runner/hdr.py`)
- [x] Brightness-weighted HDR blending
- [x] Cross-registration before blend

---

## VeraLux Processing - COMPLETE

### Stretch (`siril_job_runner/veralux_stretch.py`, `veralux_core.py`)
- [x] HyperMetric stretch with target median
- [x] Binary search for D parameter

### Silentium (`siril_job_runner/veralux_silentium.py`)
- [x] Noise suppression via SWT wavelets

### Revela (`siril_job_runner/veralux_revela.py`)
- [x] Detail enhancement via ATWT wavelets

### Vectra (`siril_job_runner/veralux_vectra.py`)
- [x] Smart saturation in LCH color space

### StarComposer (`siril_job_runner/veralux_starcomposer.py`)
- [x] Controlled star recomposition

---

## Orchestration - COMPLETE

### Job Runner (`siril_job_runner/job_runner.py`)
- [x] `JobRunner` class
- [x] `validate()` - Stage 0
- [x] `run_calibration()` - Stage 1
- [x] `run_preprocessing()` - Stage 2
- [x] `run_composition()` - Stage 3 & 4
- [x] `run()` - Full pipeline
- [x] Dry run mode support

### CLI Entry Point (`run_job.py`)
- [x] Argument parsing with argparse
- [x] `job_file` (positional) - path to JSON job file
- [x] `--validate` - validation only
- [x] `--stage {calibrate,preprocess,compose}` - run specific stage
- [x] `--dry-run` - show what would happen
- [x] `--base-path` - override base path
- [x] `--log` - write to log file
- [x] `--force` - force reprocessing
- [x] `--seq-stats` - view registration stats

---

## Testing - PARTIAL

### Existing Tests
- [x] `tests/test_logger.py`
- [x] `tests/test_fits_utils.py`
- [x] `tests/test_calibration.py`
- [x] `tests/test_job_config.py`
- [x] `tests/test_veralux_*.py` - VeraLux module tests

### Future Test Work
- [ ] Integration tests with mock Siril
- [ ] End-to-end tests with sample data

---

## File Structure

```
sirilScripts/
├── docs/
│   ├── architecture.md      # Processing workflow documentation
│   ├── code.md              # Code structure overview
│   └── todo.md              # This file
├── examples/
│   ├── example_lrgb_job.json
│   ├── example_lrgb_hdr_job.json
│   ├── example_sho_job.json
│   └── example_hoo_job.json
├── jobs/                     # User job files
├── logs/                     # Processing logs
├── siril_job_runner/
│   ├── __init__.py
│   ├── calibration.py        # Calibration master building
│   ├── calibration_paths.py  # Path resolution
│   ├── compose_broadband.py  # LRGB/RGB composition
│   ├── compose_narrowband.py # SHO/HOO composition
│   ├── composition.py        # Composition orchestration
│   ├── config.py             # Centralized configuration
│   ├── fits_utils.py         # FITS header parsing
│   ├── hdr.py                # HDR blending
│   ├── job_config.py         # Job file loading
│   ├── job_runner.py         # Pipeline orchestration
│   ├── job_schema.json       # Job file schema
│   ├── job_validation.py     # Job validation
│   ├── logger.py             # Logging utilities
│   ├── models.py             # Shared data models
│   ├── palettes.py           # Narrowband palettes
│   ├── preprocessing.py      # Frame preprocessing
│   ├── preprocessing_pipeline.py
│   ├── preprocessing_utils.py
│   ├── protocols.py          # Interface protocols
│   ├── sequence_*.py         # Sequence file parsing/analysis
│   ├── siril_*.py            # Siril operation wrappers
│   ├── veralux_*.py          # VeraLux processing modules
│   └── README.md             # Module documentation
├── tests/
│   ├── conftest.py
│   └── test_*.py
├── xisf_to_fits/             # XISF converter (separate tool)
├── run_job.py                # CLI entry point
├── settings.template.json    # Settings template
├── pyproject.toml
└── README.md
```
