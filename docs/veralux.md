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

## Upstream behavior and justified differences

Follow upstream algorithm behavior by default. A local deviation needs a reproduced
bug or a controlled comparison that establishes why it is needed. Existing code
and a subjective preference for stronger color are not sufficient justification.
Compare appearance changes on retained image data before adopting them.

The HyperMetric port follows [HyperMetric 1.5.2](https://gitlab.com/free-astro/siril-scripts/-/blob/main/VeraLux/VeraLux_HyperMetric_Stretch.py)
for the final per-channel midtone curve and automatic background check. The old
linked curve and one-pass solver changed color and contrast without evidence
justifying those differences. Paired M31, M42 and M45 comparisons support removing
them. With identical settings, automatic stretch values match upstream and output
pixels differ by less than two 16-bit steps in those comparisons.

Revela follows [Revela 1.0.2](https://gitlab.com/free-astro/siril-scripts/-/blob/main/VeraLux/VeraLux_Revela.py):
star protection strength is 1.0, structure uses the squared protection mask, and
noise estimates have the upstream minimum floor. Native-resolution crops from the
same three targets show the stronger protection chiefly reduces extra brightening
around stars. There is no demonstrated reason to retain the old strength of 0.8.

Vectra follows [Vectra 1.0.3](https://gitlab.com/free-astro/siril-scripts/-/blob/main/VeraLux/VeraLux_Vectra.py):
Lab conversion uses RGB samples directly, overlapping hue adjustments add together,
and saturation sliders use their full upstream strength. Star protection measures
Lab lightness on the 0-100 scale required by its threshold; the old 0-1 scale made
that protection ineffective. Hue distances wrap across zero correctly. On the
same three stretched RGB images, corrected output differs from upstream by less
than 0.000002 per channel, compared with differences as large as 0.149 before.

Silentium follows [Silentium 1.0.3](https://gitlab.com/free-astro/siril-scripts/-/blob/main/VeraLux/VeraLux_Silentium.py)'s
linear input requirement, direct RGB-to-Lab conversion, normalized lightness,
linear signal detection, slider strengths and chroma shadow protection. It runs
once before StarNet and stretching, on RGB plus luminance for broadband or each
mono narrowband channel. Working files retain float precision. Native M31, M42
and M45 RGB crops differ from the upstream no-star-list path by less than
0.000002 per channel. A fixed display stretch shows the processing-order change
without changing display parameters between images.

StarComposer follows [StarComposer 2.1.1](https://gitlab.com/free-astro/siril-scripts/-/blob/main/VeraLux/VeraLux_StarComposer.py)'s
mirror boundary for the small input blur, full-RGB anchor sampling stride, and
scalar-only result at negligible color grip. The old boundary rule changed edge
pixels by up to 0.021 on an M42 corner crop; the corrected result differs from
upstream by less than 0.0000002 there.

The following implementation differences remain:

- FITS units come from storage type, not the brightest pixel. Integer samples are
  divided by their type's maximum; floating-point samples retain their units.
  Outliers are clipped on a processing copy, and non-finite inputs are rejected.
  In a reproduced failure, changing one float pixel from 1.0999 to 1.1001 made the
  old upstream scale guess change the whole image. Original files are untouched.
- The existing NumPy/SciPy implementation uses analytic Lab conversion and
  float64 arithmetic; upstream Revela uses OpenCV and float32. This avoids adding
  OpenCV to production for a different numerical implementation of the same
  formulas. On the three inspected crops, the maximum RGB difference is below
  0.005 on a 0-1 scale. Giving the upstream engine the same Lab conversion reduces
  the maximum difference below 0.000002. This is measured agreement, not bitwise
  identity or a universal error bound.
- Silentium uses the upstream no-star-list path (uniform four-pixel FWHM,
  adaptive noise and chroma enabled). Its optional star-list mask is not ported.
  Upstream reads `list.lst` implicitly from the working directory and can reuse
  it without checking which image produced it. This pipeline processes several
  channels in one directory, so importing that behavior could apply another
  channel's stars. We keep the explicit no-list path rather than introduce that
  dependency. This is not parity with the GUI's default star-mask option; adding
  image-specific masks requires a separate comparison.
- For usable signal, the bounded stretch solver returns its last candidate when
  the requested target cannot be reached within the configured bounds. Upstream's inner solver can
  instead return its initial 2.0 value. Retaining the candidate respects the
  existing minimum/maximum controls rather than silently leaving their range.

These comparisons use the existing pipeline settings: target background 0.10,
IMX571 weights for the ASI2600, statistical anchor, and ready-to-use mode. These are
explicit processing choices, not a claim to reproduce the upstream GUI defaults
(target 0.20 and adaptive anchor enabled). GUI controls and scientific mode are
not ported. Changes to those settings need their own appearance comparison; they
are held fixed here so algorithm changes can be assessed independently.
