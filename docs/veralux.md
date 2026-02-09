# VeraLux Reference

The veralux modules in this project are based on VeraLux by Riccardo Paterniti.

## Original Source

https://gitlab.com/free-astro/siril-scripts/-/blob/main/VeraLux/

## Modules

| Module | Reference | Type |
|--------|-----------|------|
| veralux_stretch.py | [VeraLux_HyperMetric_Stretch.py](https://gitlab.com/free-astro/siril-scripts/-/blob/main/VeraLux/VeraLux_HyperMetric_Stretch.py) | Ported script |
| veralux_revela.py | [VeraLux_Revela.py](https://gitlab.com/free-astro/siril-scripts/-/blob/main/VeraLux/VeraLux_Revela.py) | Ported script |
| veralux_vectra.py | [VeraLux_Vectra.py](https://gitlab.com/free-astro/siril-scripts/-/blob/main/VeraLux/VeraLux_Vectra.py) | Ported script |
| veralux_silentium.py | [VeraLux_Silentium.py](https://gitlab.com/free-astro/siril-scripts/-/blob/main/VeraLux/VeraLux_Silentium.py) | Ported script |
| veralux_starcomposer.py | [VeraLux_StarComposer.py](https://gitlab.com/free-astro/siril-scripts/-/blob/main/VeraLux/VeraLux_StarComposer.py) | Ported script |
| veralux_core.py | -- | Shared statistics and I/O helpers |
| veralux_colorspace.py | -- | Shared color space conversions (RGB, XYZ, LAB, LCH) |
| veralux_wavelet.py | -- | Shared wavelet decomposition (a trous B3-spline) |

## Self-Containment Rule

The five ported script modules are intentionally self-contained. Algorithmic functions (stretch math, noise estimation, signal masks, etc.) are kept within each module even when similar logic exists elsewhere. This allows independent validation of each module against the upstream reference implementation.

Do not extract or share algorithmic code across VeraLux modules. If two modules have similar helper functions (e.g., `_compute_signal_mask` in both revela and vectra), that duplication is intentional.

The shared utility modules (`veralux_core.py`, `veralux_colorspace.py`, `veralux_wavelet.py`) provide general mathematical primitives (color space conversions, wavelet transforms, MAD statistics) that are not part of any single script's algorithm. These are fine to import from any VeraLux module.

## License

The original VeraLux code is licensed under GPL-3.0-or-later.
