# SPCA

SPCA reduces and analyzes Spitzer/IRAC time-series observations using aperture,
PSF and pixel-level photometry, detector decorrelation, and astrophysical models.
The [API documentation](https://spca.readthedocs.io) describes the package modules.

## Installation

SPCA requires Python 3.9 or newer:

```bash
git clone https://github.com/lisadang27/SPCA.git
cd SPCA
python -m pip install .
```

Optional dependencies are available as `.[GP]` for Gaussian-process models and
`.[download]` for archive downloads. For development:

```bash
python -m pip install -e '.[dev,download]'
pre-commit install
MPLBACKEND=Agg python -m pytest tests
```

Pre-commit formats Python/notebook code and checks file structure and whitespace.
Run it on specific files with `pre-commit run --files path/to/file`.

## Workflow

The driver scripts have corresponding notebooks for interactive analysis.
Configure the settings near the top of each driver for your observations.

1. Obtain Level 1 BCD images through the [download workflow](#programmatic-spitzerirac-downloads)
   or the [Spitzer Heritage Archive](https://sha.ipac.caltech.edu/applications/Spitzer/SHA/).
   Keep independent observing campaigns in separate directories.
2. Run `Make_Directory_Structure` on the campaign directory. It extracts ZIPs into
   `data/chN/r########/chN/bcd/`, preserves channel identities, and moves original
   ZIPs to `raw/`. Already-extracted downloads can be reused.
3. Run `Everything_Photometry` to extract light curves and rank aperture methods
   by scatter after boxcar smoothing at the configured width.
4. Inspect fluxes, centroids, widths and AOR boundaries with `QuickLook` before
   deciding which exposures or AORs to exclude.
5. Run `Decorrelation` with the desired detector and astrophysical models.
   `Poly#` selects a two-dimensional polynomial, `BLISS` a sensitivity map,
   and `PLD` pixel-level decorrelation. See the driver settings for combinations
   and optional terms. `MakeTables` summarizes saved fit results.

Photometry preparation loads all selected frames into memory. Large phase
curves may require substantial RAM. Sampler initialization and convergence
require inspection; long or stalled MCMC runs should be diagnosed before reuse.

## IRAC timing and legacy light curves

### Time conventions

Photometry defaults to **BMJD_TDB**: barycentric Julian date minus 2,400,000.5
in the TDB time scale. `time_format="bjd"` selects BJD_TDB;
`time_scale="utc"` selects barycentric UTC with the repaired frame cadence.
Model loaders require declared TDB products and normalize BJD_TDB to BMJD_TDB.
Custom transit epochs default to BMJD_TDB; supply `t0_format="bjd"` for BJD_TDB.
Archive epochs must have a documented TDB scale, or an independently verified
`t0_scale="tdb"` assertion.

`SPCA_TIMING_JSON` headers record format, scale, timing version, target direction
and binning convention. Preloaded arrays require matching `input_time_metadata`.
Changing that declaration does not convert the supplied values. Archive search
dates use UTC.

### Exposure midpoints and accuracy

For zero-based subarray frame `i`, the central-pixel midpoint is
`(i + 0.5)*FRAMTIME + 0.003231` seconds after its BCD start. The 3.231-ms
reset-phase adjustment follows the
[IRAC timing prescription](https://irsa.ipac.caltech.edu/data/SPITZER/docs/irac/irachpp/timing/).
Each BCD retains its own anchor, preserving command and AOR gaps. Historical
SPCA used a compressed `linspace` cadence that placed the last frame of a
2-second cube about one second early. Full frames use the nominal
`FRAMTIME/2` midpoint; pixel/Fowler readout-phase corrections remain uncalibrated.

SPCA starts with barycentric UTC `BMJD_OBS` and adds
`(ET_OBS - UTCS_OBS)/86400` to obtain TDB, following
[IRAC Handbook §6.3](https://irsa.ipac.caltech.edu/data/SPITZER/docs/irac/iracinstrumenthandbook/32/).
Do not apply another Earth-observer barycentric correction.

Supply `target_coord` to correct the requested header direction to the science
target. SPCA combines the
[header spacecraft state](https://irsa.ipac.caltech.edu/docs/knowledgebase/spitzer_general.html)
with the builtin ERFA barycentric Sun and evolves the delay linearly within
one BCD. The mission clock and orbit anchor are retained. Without coordinates,
the header-direction correction remains fixed within each BCD. Required header
state and direction consistency checks must pass, including for IER observations.

The documented clock uncertainty is typically about 20 ms and below 70 ms in the
worst case. Header rounding and orbit uncertainty also contribute. Cached JPL
Horizons comparisons at four epochs found projected state differences up to
57 ms; other observations may differ. Full-frame midpoint error is unbounded by
these checks. Subarray timing uses the central pixel; target coordinates are
static ICRS directions without proper-motion or parallax propagation. Precision
timing needs independently validated spacecraft ephemerides and readout timing.

### Binned measurements

Fluxes use **unweighted medians** in consecutive sorted row bins; the final
partial bin is retained. The representative timestamp is the median time of
the samples contributing to that flux. Rejected/NaN flux rows are excluded.
`Time std` is their population scatter in days. A missing timestamp for any
contributing sample makes both time statistics NaN. PLD bins likewise have
NaN time statistics if pixel medians use different exposure masks.

Aperture extraction preserves flux rejections. PSF/PLD retain their legacy
neighbor-median replacement during clipping, so their contributing series can
include imputed values. Centroids, widths and backgrounds are median-binned
with their own finite-value masks; aperture metadata additionally follows its
flux mask. Binned-level clipping rejects or replaces complete flux bins without
redefining their exposure membership. None of these extraction paths uses
inverse-variance-weighted binning.

Bins may cross AOR boundaries or observing gaps. Dataset-specific
`aorBreaks_AORs<tag>.txt` files record the boundaries for fitting and are reused
only with matching timing declarations and numerical boundaries (17-µs absolute
tolerance). A generic `aorBreaks.txt` is never
substituted. Review bin size and sampling near gaps,
transits and eclipses. Median time describes sampling: a single timestamp cannot
represent the temporal response of a nonlinear median during rapid variations.
Model integration must account for the contributing exposure times when needed.

### Correcting legacy aperture products

Older products can contain **BMJD_UTC** and the compressed subarray cadence.
Use `SPCA.timing_correction` with a complete BCD-header inventory and the
corresponding full-resolution/binned aperture pair. The supported schemas are
8 full-resolution columns and 16 binned columns. Run
`python -m SPCA.timing_correction --help` for inventory and correction options.

Correction first reconstructs the **historical** timestamp medians and scatter,
checking both surviving-flux and all-row conventions. It validates bin counts,
row order and correspondence of the flux products. Corrected bin timestamps
then use finite full-resolution flux contributors within those same bins.
Metadata records `historical_bin_mask`, the corrected `bin_mask`, and unchanged
bin membership. Historical reconstruction remains separate from output timing.
No flux, uncertainty, centroid, background, non-time NaN pattern or other
non-time token is changed. Empty contributor sets produce NaN time statistics.

For example, after creating a separate destination directory:

```bash
python -m SPCA.timing_correction correct \
    --headers headers.json --full legacy_full.dat --binned legacy_binned.dat \
    --bin-size 64 --target-coord 120.0 -20.0 \
    --output-full corrected/full.dat --output-binned corrected/binned.dat \
    --output-breaks corrected/aorBreaks_AORs123.txt --audit corrected/audit.json
```

This validates and reports without writing; add `--apply` to create separate
products and a hash-based JSON audit. `--time-format bjd` selects BJD_TDB.
The inventory command requires explicit science AORs and an expected BCD count.
Already-versioned inputs, ambiguous reconstruction, existing outputs and
in-place replacement are refused.

Corrected tables declare `product_role="timing_corrected"`. Standard extraction
checks intended output paths, including relocated tables detected through their
own metadata. Independent datasets can share a channel or technique directory;
marked corrected export trees remain protected.
Custom cleanup/forced-rerun drivers must call
`SPCA.timing_correction.assert_reduction_tree(path, recursive=True)` before
modifying a tree. This guard detects corrected tables, root markers and audit
directories; arbitrary external filesystem operations remain the caller's
responsibility. Re-correct original legacy files into a new destination when
changing correction settings.

## Extraction methods and channels

Ch1/ch2 have established use. Ch3/ch4 aperture extraction has synthetic coverage
and limited real-data checks: four BCDs and two short extractions, with complete
science reductions and detector-systematics validation still outstanding.
The supported paths below use native pixels (`oversamp=False`).

| Method | Geometry and settings | Validation limits |
| --- | --- | --- |
| Aperture | Ch1–ch4 subarrays and full-frame WCS crops; fixed/moving centroids, radius/edge search, RMS ranking | Synthetic tests cover all channels; limited real ch3/ch4 aperture checks |
| PSF | Shared image preparation; normalized 2D Gaussian in a fixed central 5×5 window | Synthetic output checks; full-frame and ch3/ch4 science series need validation |
| PLD | Shared preparation; fixed native 3×3 or 5×5 stamps centered at `(15,15)` | Synthetic output checks; full-frame and ch3/ch4 science performance needs validation |
| Companion | Specialized legacy two-source subarray extraction; caller-selected output directory | Full frames unsupported; initialization/bounds and ch3/ch4 performance need review |
| Frame diagnostics | Frame-position exclusions for nonoverlapping 64×32×32 cubes | Other geometries skip this diagnostic |

### Full-frame observations

Full-frame BCDs contain one 256×256 image. Supply a scalar Astropy `SkyCoord` or
an `(RA, Dec)` pair in degrees:

```python
from SPCA.Photometry_Aperture import get_lightcurve

get_lightcurve(
    basepath, aor_snippet, "ch2", planet,
    target_coord=(target_ra_deg, target_dec_deg),
    ignoreFrames=[], addStack=False, oversamp=False,
    r=[2.0, 2.5, 3.0], moveCentroids=[True, False],
)
```

`Everything_Photometry.py` exposes these coordinates through `target_coords`.
The first selected BCD's distorted celestial WCS defines a fixed 32×32 crop for
each AOR, placing the target between zero-based pixels 15 and 16. Later frames
retain those detector pixels while centroids track pointing drift. Crops use
NumPy `[y, x]` order and must fit entirely on the detector. Nearby-star masks
and full-frame correction stacks use the same crop offsets. Full-frame data
require `ignoreFrames=[]` or `None`; that option selects positions within
subarray cubes. Other positive cube lengths load, but diagnostics require 64.

### Products and limitations

New aperture, PSF and PLD directories use `ch1`–`ch4` prefixes, for example
`ch1ExactCircular2_3_movingCentroid` or `ch2PLD_3x3`. Existing ch1/ch2 directories
such as `3umExactCircular2_3_movingCentroid` and `4umExactCircular2_1` remain
readable without renaming. Aperture ranking uses exact paths recorded in
`bestPhOption.txt` and restricts selection to the requested dataset. Automatic
PSF/PLD lookup accepts both conventions and refuses ambiguity when both contain
the same dataset; use explicit product paths to select a copy. No wavelength
aliases are introduced for ch3/ch4.

`onlyBest=True` saves the aperture winner when a finite ranking exists.
New datasets may share technique directories, but existing science pairs cannot
be silently replaced. Use a separate analysis directory for another reduction,
processing version, target direction or crop configuration. No package cleanup
routine renames or removes legacy directories.

Aperture tables have 8/16 full/binned columns; the full table omits the internal
aperture-flux uncertainty. PSF tables have 7/14 columns; PLD tables contain pixel
vectors plus time/background columns. Companion full tables include flux
uncertainty. Binned `Flux std` is empirical population scatter, without a formal
uncertainty on the median. Model loaders retain method-specific schemas and
require matching time rows when combining products.

All channels use their own header calibration values. Aperture centroid and
background-exclusion windows are configurable; assess them for broad Si:As PSFs
and pointing drift. Detector ramps, persistence, intrapixel sensitivity,
[banding](https://irsa.ipac.caltech.edu/data/SPITZER/docs/irac/iracinstrumenthandbook/40/)
and [aperture corrections](https://irsa.ipac.caltech.edu/data/SPITZER/docs/irac/iracinstrumenthandbook/17/)
require observation-specific assessment.

The automatic stellar brightness-temperature approximation uses
[PHOENIX-ACES HiRes files](https://phoenix.astro.physik.uni-goettingen.de/) covering
approximately 0.05–5.5 µm. They cannot span the ch3/ch4 response functions.
Extending this calculation requires stellar spectra covering those complete
bandpasses and response integration. This limit belongs to the selected files;
other PHOENIX products may have different coverage. The existing ch1/ch2
calculation compares spectral sample sums over broad 3–4/4–5 µm windows with
Planck spectra. It checks that loaded spectra span those windows and have finite
samples, but remains an approximation without IRAC response weighting or
wavelength quadrature. Ch3/ch4 analysis drivers that invoke it raise an error.

**Oversampling is currently nonfunctional.** Leave `oversamp=False` until the
interpolator's runtime and axis-order defects are fixed and scientifically
validated. PSF fits also assume unit sampling scale; the separate legacy
companion interpolator remains unvalidated. Oversampling caches identify the
full archive stem, scale, prepared pixels and masks; malformed entries regenerate,
and old sequence-number caches are ignored. Any interpolator repair must bump
`OVERSAMPLE_CACHE_VERSION` to prevent reuse of old results.

Image preparation derives masks from science pixels and does not consume
archive uncertainty/mask companion files. `addStack=True` selects external
`Calibration/` correction stacks using each AOR's sky-dark filename. The
sky-dark pixels are not used, and SPCA does not supply the external stacks.

## Programmatic Spitzer/IRAC downloads

Install `.[download]` and use `Download_Spitzer_Data.ipynb` or
`python Download_Spitzer_Data.py --help`. Both share `SPCA/spitzer_download.py`.
The default performs metadata discovery only. Explicit coordinates bypass name
resolution; decimal degrees and quoted sexagesimal values are accepted.
Negative declinations can use `--dec=-12:34:56`.

```bash
# Discover candidate observations; dates are UTC.
python Download_Spitzer_Data.py \
    --ra 06:30:32.797 --dec +29:40:20.26 --channels 1 2 3 4 \
    --start 2008-01-01 --end 2008-12-31T23:59:59.999 \
    --output /path/to/campaign

# Preview selected AORKEYs from discovery without transferring FITS files.
python Download_Spitzer_Data.py \
    --ra 06:30:32.797 --dec +29:40:20.26 --aor AORKEY1 AORKEY2 \
    --channels 1 2 --output /path/to/campaign --preview

# Download or restart the saved plan.
python Download_Spitzer_Data.py \
    --manifest /path/to/campaign/download_plan.ecsv --download
```

`--pid`, `--aor`, dates and channel filters intersect; values within a list are
alternatives. Date filters select whole overlapping AORs. Discovery and planning
expand AOR time coverage, which can take minutes for long observations.
A saved manifest fixes its profile and science version; regenerate the plan
to change either selection.

### Product selection

| Profile | Requested products |
| --- | --- |
| `spca` (default) | Selected science images and `cal/*_sdark.fits` for optional correction-stack lookup |
| `science-qa` | Above plus same-exposure uncertainty and supported mask products |
| `archive` | Supported Level 1 `bcd/` and `cal/` FITS products, including both BCD and CBCD |

`--science-product bcd` is the default; `cbcd` selects artifact-corrected images.
Only the selected version counts as science. Its absence is an error. SPCA
keeps alternate versions separate and does not generate CBCD artifact corrections.
The extraction APIs accept `science_product="cbcd"` for this choice. Small real
ch3/ch4 checks used BCDs; CBCD science equivalence has not been established.
See the [IRAC product table](https://irsa.ipac.caltech.edu/data/SPITZER/docs/irac/iracinstrumenthandbook/31/).

Raw/Level 2 data, logs and unsupported products are excluded and reported in the
preview. QA selection skips orphan files without corresponding selected science
images. Shared calibration products are stored under each associated science
AOR. Changing profile/version preserves verified existing files and their
inventory records.

### Completeness and verification

SIA finds exposures whose footprints overlap the search circle. IRAC ch1/ch3
share one field and ch2/ch4 another, so a positional search can omit the adjacent
field. Successful transfer alone leaves archive completeness unverified.
Use `--compare` for the archive directory listing and quality-report counts,
or `--reference /path/to/SHA_download.zip` for an independently obtained ZIP or
directory. Comparisons cover `bcd/` products and report missing, extra,
intentionally excluded and unplanned-channel files. An unparseable listing is
an error. Optional QA association counts do not establish completeness.

Transfers validate HTTP lengths and FITS structure without rewriting image data.
Local SHA-256 hashes verify completed downloads; IRSA supplies no archive
checksums. Interrupted files restart from zero. Atomic inventory checkpoints
and a per-file journal recover verified progress after interruption. Untracked
existing files cause a conflict; replacing them requires
`--overwrite-untracked`. Corrupted inventory-tracked files are repaired.

Campaign bookkeeping includes discovery, plan, manifest, inventory and
completeness ECSV files, a transient `download_journal.jsonl`, and an optional
`inventory_comparison.ecsv`. Science files are written directly to
`data/chN/r########/chN/bcd/`. CLI downloads run directory setup automatically;
notebook users run the final setup cell. Configure `Everything_Photometry` with
the campaign and selected AORs afterward. An AOR prefix alone is not a date filter.

References: [IRSA APIs](https://irsa.ipac.caltech.edu/voapi.html),
[image APIs](https://irsa.ipac.caltech.edu/docs/program_interface/api_images.html),
[SHA knowledgebase](https://irsa.ipac.caltech.edu/docs/knowledgebase/spitzer_sha.html),
and [IRSA tutorials](https://caltech-ipac.github.io/irsa-tutorials/).

## Validation

Tests run offline using synthetic images, six real header records and cached
Horizons vectors. They cover timing, correction safeguards, channel routing,
WCS extraction, aperture masks and download bookkeeping. They do not establish
full ch3/ch4 science calibration. Optional local-data tools accept arbitrary
targets: `python tools/validate_irac_samples.py --help` and
`python tools/check_irac_ephemeris.py --help`.

## Contributors and acknowledgements

[Lisa Dang](https://github.com/lisadang27) created SPCA and developed much of its
photometry, fitting, polynomial and PLD routines.
[Taylor James Bell](https://github.com/taylorbell57) generalized the pipeline,
improved its documentation and contributed GP decorrelation. Both authors
worked on debugging and usability. We thank Joel Schwartz for BLISS guidance,
Dylan Keating for testing, and Antoine Darveau-Bernier for the
[Exoplanet Archive parameter-selection code](https://github.com/AntoineDarveau/masterfile).

## License and citation

Copyright © 2018–2020 Lisa Dang and Taylor James Bell. SPCA is distributed under
the [GPL3 License](./LICENSE). Please cite
[Dang et al. (2018)](https://arxiv.org/abs/1801.06548) when using SPCA
([ADS](https://ui.adsabs.harvard.edu/abs/2018NatAs...2..220D),
[BibTeX](https://ui.adsabs.harvard.edu/abs/2018NatAs...2..220D/exportcitation)).
