# Implementation Todo List

## Code Standards

- **<300 LOC per file** — split if approaching limit
- **Clear separation of concerns** — one responsibility per module
- **Tests via pytest** — test each module independently
- **High quality, not over-engineered** — internal single-user tool
- **No unnecessary abstractions** — keep it simple and direct

---

## Phase 1: Core Infrastructure

### 1.1 Logger (`src/logger.py`)
- [ ] Create `JobLogger` class
- [ ] Implement elapsed time tracking `[MM:SS]` format
- [ ] Console output with indentation levels
- [ ] Log file creation at `{output}/job_log_{timestamp}.txt`
- [ ] Methods: `info()`, `step()`, `substep()`, `warning()`, `error()`
- [ ] Context manager for timed operations

### 1.2 FITS Utilities (`src/fits_utils.py`)
- [ ] Function to read single FITS header
- [ ] Extract: exposure time, sensor temperature, filter name
- [ ] Handle common FITS keyword variations (EXPTIME vs EXPOSURE, etc.)
- [ ] Function to scan directory of FITS files
- [ ] Build requirements table from list of files
- [ ] Return unique (exposure, temp, filter) combinations with counts

### 1.3 Job Schema (`src/job_schema.json`)
- [ ] Define JSON schema for job files
- [ ] Required fields: name, type, calibration, lights, output
- [ ] Optional fields: options (fwhm_filter, temp_tolerance, denoise)
- [ ] Validation for type enum: LRGB, SHO, HOO
- [ ] Validation for calibration dates format

---

## Phase 2: Calibration System

### 2.1 Calibration Module (`src/calibration.py`)
- [ ] `CalibrationManager` class
- [ ] Constructor takes base path and calibration dates
- [ ] Method: `find_master(type, params)` - returns path or None
- [ ] Method: `can_build_master(type, params)` - checks raw frames exist
- [ ] Method: `build_master(type, params, siril)` - stacks and caches
- [ ] Temperature tolerance matching (±2°C default)
- [ ] Master path resolution:
  - [ ] `masters/bias/bias_{temp}_{date}.fit`
  - [ ] `masters/darks/dark_{exposure}_{temp}_{date}.fit`
  - [ ] `masters/flats/flat_{filter}_{date}.fit`
- [ ] Raw path resolution:
  - [ ] `raw/bias/{date}_{temp}/`
  - [ ] `raw/darks/{date}_{exposure}_{temp}/`
  - [ ] `raw/flats/{date}/{filter}/`

### 2.2 Calibration Building
- [ ] Build bias master: `stack rej 3 3 -nonorm`
- [ ] Build dark master: `stack rej 3 3 -nonorm`
- [ ] Build flat master: `calibrate` with bias, then `stack rej 3 3 -norm=mul`

---

## Phase 3: Preprocessing

### 3.1 Preprocessing Module (`src/preprocessing.py`)
- [ ] `Preprocessor` class
- [ ] Method: `process_filter(filter_name, light_paths, calibration, output_dir)`
- [ ] Steps:
  - [ ] `convert` - Load raw lights into sequence
  - [ ] `calibrate` - Apply bias, dark, flat
  - [ ] `seqsubsky` - Background extraction (degree 1)
  - [ ] `register` - 2-pass registration
  - [ ] `seqapplyreg` - Apply with FWHM filter
  - [ ] `stack` - Winsorized sigma rejection (`rej w 3 3 -norm=addscale`)
- [ ] Return path to stacked result
- [ ] Handle multi-directory light sources (multi-night)

---

## Phase 4: Composition

### 4.1 Composition Module (`src/composition.py`)
- [ ] `Composer` class
- [ ] Method: `compose_lrgb(stacks_dict, output_dir)`
  - [ ] Register stacks across filters
  - [ ] Linear match R, G, B to L (or reference)
  - [ ] Optional: deconvolution on L
  - [ ] `rgbcomp R G B`
  - [ ] `rgbcomp -lum=L rgb`
- [ ] Method: `compose_sho(stacks_dict, output_dir, palette)`
  - [ ] Register stacks
  - [ ] Linear match to H
  - [ ] Palette mixing via pixelmath
  - [ ] Palettes: HOO, SHO_blue_gold, SHO_blue_red
- [ ] Method: `compose_hoo(stacks_dict, output_dir)`
  - [ ] Simplified HOO composition
- [ ] Save unstretched output

### 4.2 Stretching (`src/composition.py`)
- [ ] Method: `stretch(input_path, output_dir)`
- [ ] Save unstretched copy first
- [ ] Apply: `autostretch`, `mtf 0.20 0.5 1.0`, `satu 1 0`
- [ ] Save outputs: `.fit`, `.tif` (astro deflate), `.jpg` (90%)
- [ ] Add VeraLux placeholder comment

---

## Phase 5: Orchestration

### 5.1 Job Runner (`src/job_runner.py`)
- [ ] `JobRunner` class
- [ ] Constructor: load and validate job file
- [ ] Method: `validate()` - Stage 0
  - [ ] Scan all light frame headers
  - [ ] Build requirements table
  - [ ] Check calibration availability
  - [ ] Return validation report
- [ ] Method: `run_calibration()` - Stage 1
  - [ ] Build any missing masters
- [ ] Method: `run_preprocessing()` - Stage 2
  - [ ] Process each filter
- [ ] Method: `run_composition()` - Stage 3 + 4
  - [ ] Compose based on type
  - [ ] Stretch
- [ ] Method: `run()` - Full pipeline
- [ ] Dry run mode support

### 5.2 CLI Entry Point (`run_job.py`)
- [ ] Argument parsing with argparse
- [ ] Arguments:
  - [ ] `job_file` (positional) - path to JSON job file
  - [ ] `--validate` - validation only
  - [ ] `--stage {calibrate,preprocess,compose}` - run specific stage
  - [ ] `--dry-run` - show what would happen
  - [ ] `--base-path` - override base path (default from job or cwd)
- [ ] Initialize sirilpy connection
- [ ] Run JobRunner
- [ ] Handle errors gracefully

---

## Phase 6: Testing

### 6.1 Test Infrastructure
- [ ] Create `tests/` directory
- [ ] Create `tests/conftest.py` with shared fixtures
- [ ] Create sample FITS headers for testing (mock data)

### 6.2 Unit Tests
- [ ] `tests/test_logger.py`
  - [ ] Test elapsed time formatting
  - [ ] Test log file creation
  - [ ] Test different log levels
- [ ] `tests/test_fits_utils.py`
  - [ ] Test header extraction
  - [ ] Test keyword variations
  - [ ] Test requirements table building
- [ ] `tests/test_calibration.py`
  - [ ] Test master path resolution
  - [ ] Test temperature tolerance matching
  - [ ] Test can_build_master logic
- [ ] `tests/test_job_runner.py`
  - [ ] Test job file validation
  - [ ] Test validation stage output

### 6.3 Integration Tests
- [ ] Test with real calibration data
- [ ] Test with multi-night lights
- [ ] Test full pipeline end-to-end

---

## Phase 7: Polish

### 7.1 Example Job File
- [ ] Create `examples/example_lrgb_job.json`
- [ ] Create `examples/example_sho_job.json`

### 7.2 Documentation
- [ ] Update architecture.md if needed
- [ ] Add usage examples to README

---

## File Structure

```
sirilScripts/
├── docs/
│   ├── architecture.md
│   └── todo.md
├── src/
│   ├── logger.py          (<300 LOC)
│   ├── fits_utils.py      (<300 LOC)
│   ├── calibration.py     (<300 LOC)
│   ├── preprocessing.py   (<300 LOC)
│   ├── composition.py     (<300 LOC)
│   ├── job_runner.py      (<300 LOC)
│   └── job_schema.json
├── tests/
│   ├── conftest.py
│   ├── test_logger.py
│   ├── test_fits_utils.py
│   ├── test_calibration.py
│   └── test_job_runner.py
├── examples/
│   ├── example_lrgb_job.json
│   └── example_sho_job.json
├── run_job.py
├── pytest.ini
└── README.md
```

---

## Implementation Order

1. `src/logger.py`
2. `src/fits_utils.py`
3. `src/calibration.py`
4. `src/job_schema.json`
5. `src/preprocessing.py`
6. `src/composition.py`
7. `src/job_runner.py`
8. `run_job.py`
9. `examples/example_job.json`
