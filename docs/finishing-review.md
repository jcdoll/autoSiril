# Finishing comparison

The shared HyperMetric target is 0.12. The upstream Curves RGB/K tool is available
for job-specific adjustments, with an identity curve by default. No supplied job
enables a curve: the comparisons show that one stronger curve is not suitable for
every target. M31/M42 retain 0.15; M33/NGC4244 retain 0.10.

## What changed and why

- Narrowband now finishes the starless image before saving it and restoring
  stars. Previously the starless export missed saturation and detail finishing,
  while the recombined image received them. Broadband already used this order.
- HyperMetric, Revela, Vectra and StarComposer now save float32 working FITS,
  avoiding repeated 16-bit rounding between stages. Final JPEG export is unchanged.
- Requested background extraction, channel matching, deconvolution and finishing
  failures stop composition. A failed VeraLux stretch no longer produces an
  autostretch image with a VeraLux filename.
- FITS clipping diagnostics recognize big-endian floating-point storage, as well
  as native byte order, so valid float pixels are not counted as 16-bit integers.

## Image evidence

Full-resolution finishing replays cover M45, M31, M42, IC2118, NGC4244 and IC1396.
They use retained stretched RGB/L or narrowband channels, real Siril composition
and export, and the production finishing functions. Three views compare the
current library, the proposed baseline and that baseline plus the same optional
curve. Overviews and fixed native-pixel subject/background crops have identical
display scaling; no automatic display stretch is applied.

| Target | Observed result | Decision |
|--------|-----------------|----------|
| M45 | 0.12 gives a darker sky; the curve also emphasizes grain and bright-star halos | Keep 0.12 and identity curve |
| IC2118 | 0.12 reduces the gray background; the curve gives a modest lift with existing grain and halos | Keep 0.12 and identity curve |
| IC1396 | 0.12 darkens the sky; the curve makes nebula structure more visible with little change in the selected dark region | Show both for user review; identity remains default |
| M31 | 0.15 remains stable; the curve brightens and broadens the already bright center | Keep 0.15 and identity curve |
| M42 | 0.15 remains stable; the curve makes the bright center more washed out | Keep 0.15 and identity curve |
| NGC4244 | 0.10 remains stable; the curve modestly brightens the small galaxy and stars | Keep 0.10 and identity curve |

For IC1396, measured median display brightness in one fixed dark region is
31.35 without the curve and 31.42 with it, on a 0-255 scale. The fixed subject
region rises from 67.34 to 79.60. These are presentation measurements, not
signal-to-noise estimates or proof of recovered detail. Native crops still show
grain, and M45's existing thin green trail remains visible.

Separate IC1396 pairs hold the stretched pixels fixed and change only finishing
order, for both VeraLux and autostretch. The starless export now receives the
intended color finishing. The recombined views remain close in appearance.

The optional curve used for these comparisons can be reproduced in job options:

```json
{
  "veralux_curves_points": [[0, 0], [0.12, 0.12], [0.3, 0.36], [0.55, 0.61], [0.8, 0.8], [1, 1]],
  "veralux_curves_range": [0.14, 0.7],
  "veralux_curves_feather": 0.04
}
```

## Palette experiment not adopted

IC1396 and SH2-184 also compare stretching mono channels before SHO assembly
against assembling linear SHO before HyperMetric. Both use target 0.20 and the
same retained linear starless channels, existing normalization and finishing.
SH2-184 keeps the same stretched luminance in both versions. Linear assembly
followed by stretching produces a strong purple background and greener emission
in these tests; it does not justify changing the production order.

This is an ordering experiment, not a port of
[VeraLux Alchemy](https://gitlab.com/free-astro/siril-scripts/-/blob/main/VeraLux/VeraLux_Alchemy.py).
Alchemy's mixer derives two signals, Ha from red and OIII from green/blue, including
its pseudo-SHO preset. That is not a direct replacement for three independent
SII, Ha and OIII filter images. A different palette workflow needs its own color
comparison rather than an incidental change in this cleanup.

## Validation limits

Retained checkpoints isolate finishing changes; these trials do not rerun raw
calibration, stacking, SPCC or StarNet. The palette trial exercises HyperMetric on
full-resolution linear RGB, and the autostretch trial starts from linear mono
channels. Curves tests use independent upstream output fixtures; composition
tests exercise failed requested stages and starless finishing through public
entry points. Precision tests check faint-signal retention and FITS metadata.

The current-library panels include earlier processing and JPEG encoding choices,
so they do not attribute every difference to one setting. All source files read
by the trials and all 81 current output JPEGs were checked by SHA256 for changes.
The library is unchanged; comparison images are separate review artifacts.
