# Image-quality corrections

These job settings follow comparisons of the actual output images. They do not
change the upstream VeraLux algorithms. PR #11 contains overview and native-pixel
comparisons; original captures and previous exports are preserved.

## M42

M42 disables deconvolution. With the same luminance input and Silentium settings,
deconvolution introduced dark undershoots around small stars. The corrected full
render removes those rings at the cost of slightly softer stars.

The job also inherits the default autostretch white point of 1.0 instead of 0.95.
In the isolated luminance comparison, the 0.95 setting clipped 82,705 pixels;
changing only that setting to 1.0 reduced the count to one. With deconvolution
also disabled, none reached white in that luminance checkpoint. The final
comparison shows more structure in the bright core. This is not a claim that
all final RGB channels or original exposures are free of saturation.

## IC2118

The large magenta halo and thin vertical trail were present before VeraLux
processing. Individual registered exposures showed varying broad halos and
moving trails. Stronger rejection thresholds, MAD and generalized ESD did not
remove the trail as cleanly as excluding the contaminated exposures.

The job now lists individual retained files for affected nights, keeping whole
directories for unaffected groups. It omits 16 R, 6 B and 12 L captures from
processing; every original remains on disk. Input counts before automatic
FWHM rejection are:

| Filter | Previous | Selected |
| --- | ---: | ---: |
| L | 47 | 35 |
| R | 50 | 34 |
| G | 48 | 48 |
| B | 30 | 24 |

The controlled restack used the existing Winsorized rejection settings, 3/3.
The right-hand magenta ring was dramatically reduced, and the vertical trail
was no longer visible in the inspected crop. Omitting exposures reduces total
integration time and can increase noise; stronger smoothing is not used to hide
that tradeoff. The final job must be run through preprocessing after selection
changes; compose-only uses the already retained stacks.

The job also enables the existing green-cast removal setting. The starless
branch retained a green glow around the brightest stars that became conspicuous
when stars were restored. A full-image trial removes that green tint in both
autostretch and VeraLux outputs while retaining the blue reflection nebula.
It also shifts background hues toward blue, brown and purple; the underlying
bright-star halo becomes warmer rather than disappearing geometrically.
Reducing StarComposer color grip did not fix it, so that setting stays unchanged.
Some upper-edge purple glow remains; the available evidence does not justify
removing broad background structure that may include real faint nebulosity.

## Color-calibration checks

SPCC background tolerances were passed in the wrong order. Siril expects
`lower,upper`; the corrected command produces the configured interval of
-2.8 to +2.0 sigma. The lower setting is a positive magnitude in this program;
Siril accepts that magnitude and applies the negative sign.
[SPCC command reference](https://siril.readthedocs.io/en/stable/processing/color-calibration/spcc.html)

When SPCC is enabled, a failed plate solve or SPCC command now stops the job.
It no longer exports an uncalibrated image as a successful run. Disabling SPCC
explicitly remains supported. Successful commands retain Siril's own warnings.

Siril 1.4.3 still reports an imprecise solution on these broadband images. Its
[fit diagnostic](https://github.com/lock042/siril/blob/1.4.3/src/algos/fitting.c#L161-L198)
sums squared residuals from all stars but divides by the inlier count, making
the reported scatter sensitive to outliers. This affects the diagnostic; it
does not itself prove that a particular calibration is accurate.

Independent brighter-star subsets changed the applied color coefficients by
less than 0.8% for M31, up to 1.6% for M42, and up to 1.8% for LDN1082, although
the reported scatter sometimes increased. Changing the photometry aperture
setting did not change the coefficients because this Siril version forces
dynamic apertures internally. These checks do not justify changing the
production photometry settings or suppressing the warning. IC2118's subset
trial hit a catalog HTTP 429 error and an upstream Siril crash; it provided no
calibration-stability result.
