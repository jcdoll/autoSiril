# autoSiril

Automation tools for astrophotography image processing.

## Sample Output

<p align="center">
  <a href="images/M31.jpg"><img src="images/M31.jpg" width="45%" alt="M31"></a>
  <a href="images/M33.jpg"><img src="images/M33.jpg" width="45%" alt="M33"></a>
</p>
<p align="center">
  <a href="images/M45.jpg"><img src="images/M45.jpg" width="45%" alt="M45"></a>
  <a href="images/SH2-190.jpg"><img src="images/SH2-190.jpg" width="45%" alt="SH2-190"></a>
</p>

## Projects

### Siril Job Runner

Automated Siril processing pipeline with JSON job file configuration.

```bash
# Run a single job
uv run run-job jobs/M42.json

# Run multiple jobs
uv run run-job jobs/M42.json jobs/M31.json

# Run all jobs in a directory
uv run run-job jobs/

# Run jobs matching a pattern
uv run run-job jobs/ --pattern "SH2*"

# Run all jobs with logging and continue after failures
uv run run-job jobs/ --log --continue-on-error
```

Successful full jobs and `--stage compose` runs copy final output files and job logs into `<base_path>/outputs/<target>_<type>_output`. The target-level `processed*` folders remain disposable working folders and can be cleaned after outputs are archived.

Features:
- Job file-based configuration for reproducible processing
- Auto-detection of calibration requirements from FITS headers
- Temperature tolerance matching for darks/bias
- Multi-night light frame support
- Support for LRGB, SHO, and HOO workflows
- VeraLux processing steps: Stretch, Silentium (noise), Revela (detail), Vectra (saturation), StarComposer

See [siril_job_runner/README.md](siril_job_runner/README.md) for full documentation.

### XISF to FITS Converter

Batch convert XISF files to FITS format.

```bash
# Convert all XISF files, excluding processed folders
uv run python -m xisf_to_fits /path/to/images -e process
```

Features:
- Recursive directory scanning
- Exclude patterns for processed/calibration folders
- Progress bar with ETA
- Verification of converted files

See [xisf_to_fits/README.md](xisf_to_fits/README.md) for full documentation.

## Installation

Requires [uv](https://docs.astral.sh/uv/) for dependency management.

```bash
git clone https://github.com/yourusername/autoSiril.git
cd autoSiril
uv sync
```

## Development

Requires Python 3.11+.

```bash
# Run tests
uv run pytest

# Run tests with coverage
uv run pytest --cov=siril_job_runner --cov-report=term-missing

# Lint and format
uvx ruff check siril_job_runner/ tests/ --fix --extend-select I,B,SIM,C4,ISC,PIE && uvx ruff format siril_job_runner/ tests/
```

CI runs automatically on push to main and on pull requests via GitHub Actions.
