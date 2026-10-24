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

## Linear star masks and FITS units

StarComposer uses the bounded rational curve from upstream StarComposer 2.1.1.
Its input is a linear star mask, so no gamma conversion or fixed faint-star
cutoff is applied. Both broadband and narrowband composition use this path;
a mono narrowband mask produces white stars. The existing intensity control
now uses the upstream 1-21 range, with a pipeline default of 5 and neutral
hardness 50. M42 uses intensity 11 because its RGB star mask is much fainter
than the mono narrowband masks; the side-by-side comparison retains a visible
star field without overwhelming the nebula. Old intensity overrides used a different curve and must be retuned.
The GUI and optional upstream healing/reduction features are not ported.

Shared FITS I/O converts integer samples using their dtype's full scale and
keeps floating-point samples in their existing units. Pixels outside [0, 1]
are clipped at the processing boundary, never used to infer image units.
This prevents a few StarNet outliers from darkening an entire image. Original
star-mask checkpoints are kept unchanged for diagnosis.

The seven supplied narrowband jobs disable whole-channel background matching.
Their October comparisons showed that fitting green and blue to red suppressed
nebula structure. This is a job setting, not a new palette or a global saturation
change. Compare the final images before adjusting color further.

Narrowband saves its already-stretched checkpoint as `*_palette_<method>.fit`; it does not report it as an unstretched RGB image.
Use the retained filter stacks for manual linear narrowband processing.
LRGB reports the existing `rgb_linear.fit` as its linear RGB product.

## License

The original VeraLux code is licensed under GPL-3.0-or-later.

## Differences from upstream

The HyperMetric port retains a luminance-linked final midtone adjustment. Upstream
[HyperMetric 1.5.2](https://gitlab.com/free-astro/siril-scripts/-/blob/main/VeraLux/VeraLux_HyperMetric_Stretch.py)
applies that adjustment separately to each color channel. These produce different
colors; the local implementation is not an exact reference match.

Image loading uses the storage type to determine units. Integer images are divided
by their type's maximum; floating-point pixels retain their units. Out-of-range
pixels are clipped on the processing copy, and non-finite inputs are rejected.
This deliberately avoids the upstream maximum-based scale guess.

Revela squares the protection mask for stronger protection during structure
enhancement, following the upstream formula. Its existing protection strength
remains 0.8; upstream Revela 1.0.2 uses 1.0. Changing that strength is a separate
appearance decision.
