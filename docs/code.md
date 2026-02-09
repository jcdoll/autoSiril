# Code Structure

Orientation guide for the Siril Job Runner codebase.

## Repository Layout

```
autoSiril/
├── .github/workflows/ci.yml    # GitHub Actions CI (lint, format, test)
├── docs/                        # Documentation
├── examples/                    # Example job files
├── jobs/                        # User job files
├── logs/                        # Processing logs
├── siril_job_runner/            # Main package
├── tests/                       # pytest tests
├── xisf_to_fits/                # Separate XISF converter tool
├── run_job.py                   # CLI entry point
├── settings.json                # User settings (git-ignored)
└── settings.template.json       # User settings template
```

## Core Concepts

- Job files are JSON configurations defining what to process (target, filters, calibration dates, options)
- `Config` dataclass in `config.py` holds all processing parameters with `StrEnum` types for closed-set fields; users override via `settings.json` or job options
- `SirilWrapper` in `siril_wrapper.py` wraps pysiril; all Siril operations go through this interface
- Configuration enums (`StretchMethod`, `ColorRemovalMode`, `SubskyMethod`, `Channel`, etc.) are defined in `config.py` and provide validation at load time

## Processing Pipeline

```
run_job.py -> JobRunner
                |
                +-- 1. Validation: scan FITS headers, check calibration
                |
                +-- 2. Calibration: build missing masters (bias, dark, flat)
                |
                +-- 3. Preprocessing: calibrate -> register -> stack (per filter+exposure)
                |
                +-- 4. Composition: combine stacks -> stretch -> enhance -> save outputs
```

## Module Groups

| Group | Purpose | Key Files |
|-------|---------|-----------|
| Orchestration | Pipeline control | `job_runner.py`, `job_config.py`, `job_validation.py` |
| Configuration | Defaults, enums, overrides | `config.py` (StrEnum types, Config dataclass) |
| Calibration | Master frame building | `calibration.py`, `calibration_paths.py` |
| Preprocessing | Frame processing | `preprocessing.py`, `preprocessing_pipeline.py`, `preprocessing_utils.py` |
| Composition | Image combination | `composition.py`, `compose_broadband.py`, `compose_narrowband.py`, `compose_helpers.py` |
| VeraLux | Stretch and enhancement | `veralux_stretch.py`, `veralux_silentium.py`, `veralux_revela.py`, `veralux_vectra.py`, `veralux_starcomposer.py` |
| VeraLux Utilities | Shared math | `veralux_core.py`, `veralux_colorspace.py`, `veralux_wavelet.py` |
| Siril Interface | Command wrappers | `siril_wrapper.py`, `siril_file_ops.py`, `siril_registration.py`, `siril_stretch.py`, `siril_color.py` |
| Analysis | Frame and sequence stats | `fits_utils.py`, `frame_analysis.py`, `sequence_parse.py`, `sequence_stats.py`, `sequence_threshold.py`, `sequence_analysis.py`, `psf_analysis.py` |
| Support | Data models, logging | `models.py`, `logger.py`, `protocols.py` |
| Narrowband | Palette and stack tools | `palettes.py`, `stack_discovery.py` |
| HDR | Multi-exposure blending | `hdr.py` |

## Key Entry Points

- `run_job.py` - CLI, parses args, loads job, runs pipeline
- `JobRunner` in `job_runner.py` - orchestrates all stages
- `compose_and_stretch()` in `composition.py` - routes to broadband/narrowband handlers

## Configuration

All defaults live in `Config` dataclass (`config.py`). Override precedence:

```
defaults -> settings.json -> job file options
```

Closed-set string fields use `StrEnum` types (e.g., `StretchMethod`, `SubskyMethod`, `BlendMode`). JSON string values are coerced to enums at load time, providing immediate validation of typos.

See `config.py` for available options; `job_schema.json` for job file format.

## VeraLux Module Design

VeraLux modules are ported from standalone reference scripts by Riccardo Paterniti. Algorithmic functions are intentionally self-contained to allow independent validation against the upstream source. Do not extract shared algorithmic code across VeraLux modules.

Shared math utilities (color space conversions in `veralux_colorspace.py`, wavelet transforms in `veralux_wavelet.py`) are not part of any single script's algorithm and are shared normally.

## Post-Stretch Pipeline (Broadband)

After any stretch method (autostretch or veralux), broadband images go through the same post-processing pipeline via `_apply_post_stretch` in `compose_broadband.py`:

1. Color cast removal (SCNR)
2. Background color neutralization
3. Saturation adjustment
4. VeraLux enhancements (Silentium, Revela, Vectra)

The stretch method only affects tone mapping. Post-stretch processing is always the same regardless of which stretch was used.
