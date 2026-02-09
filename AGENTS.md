# Additional Conventions Beyond the Built-in Functions

As this project's AI coding tool, you must follow the additional conventions below, in addition to the built-in functions.

# Overview

Engineering tools for internal use.

## Repository layout

- `siril_job_runner/` - main Python package (Siril processing pipeline)
- `tests/` - pytest tests
- `jobs/` - user job JSON files
- `examples/` - example job files
- `docs/` - architecture, code structure, VeraLux reference
- `xisf_to_fits/` - separate XISF-to-FITS converter tool
- `run_job.py` - CLI entry point
- `settings.json` / `settings.template.json` - user settings
- `.github/workflows/ci.yml` - GitHub Actions CI

## Tech Stack

- Python (>=3.11)
	- Environment: `uv`
	- Lint: `ruff`
	- Test: `pytest`
	- Type checking: `ty`
	- Build: `hatchling`
	- CI: GitHub Actions
- Matlab
	- Lint: `checkcode`
	- Test: `runtests`
- KLayout
	- pcell: use python not ruby

## Commands

- Python
	- Install: `uv sync` from pyproject.toml
	- Lint: `uvx ruff check [path] --fix --extend-select I,B,SIM,C4,ISC,PIE && uvx ruff format [path]`
	- Test: `uv run pytest`
	- Test with coverage: `uv run pytest --cov=siril_job_runner --cov-report=term-missing`
	- Type check: `uvx ty check [path]`
- Matlab
	- Test: `matlab -batch "runtests('tests')"

## Safety

- Never run `git checkout` (destructive)
- Never use unicode in code (breaks Python on Windows)

## Workflow

- Plan first, share plan, get explicit approval before implementing
- Never assume a good plan means you should implement it
- If unsure or stuck, ask the user
- If something fails, do not silently move on - ask for clarification
- If `uv` fails due to permissions, consult user

## Testing

- Run tests after changes; all tests must pass before task is complete
- Prefer tests over manual verification

## Code Quality

- Use type hints and docstrings for all functions
- Target files under 300 LOC, but don't split artificially (inherent domain complexity is acceptable)
- Fix all linter issues

## Parameters

- Single source of truth for defaults (e.g. config.py)
- Use enums instead of bare strings
- Never use `params.get('key', default)` pattern
- Never use `if config then value else default` pattern

## Refactoring

- Delete old interfaces; no legacy wrappers or thin compatibility layers

## VeraLux Modules

- VeraLux modules (`veralux_stretch`, `veralux_revela`, `veralux_vectra`, `veralux_silentium`, `veralux_starcomposer`) are ported from standalone reference scripts
- Algorithmic functions in these modules are intentionally self-contained to allow independent validation against the upstream source
- Do not extract or share algorithmic code across VeraLux modules
- Shared math utilities (color space conversions, wavelet transforms) live in `veralux_colorspace.py` and `veralux_wavelet.py` and are fine to share

## Documentation

- No emojis in code or docs
- No excessive bold in markdown; use styling selectively
- No "last updated" dates or authorship
- Commands on single line, not split across lines

## Hardware

- Read project-specific docs and datasheets before making suggestions
- Use exact part numbers, not generic equivalents
- Prioritize safety over convenience; err on side of caution

## Plotting

- For python, include `addcopyfighandler` for easy matplotlib copy/paste
