"""IRAC absolute barycentric timestamps, in BMJD_TDB by default.

JD/MJD is a format; TDB/UTC is a scale; neither changes the sight line.
BMJD_OBS is barycentric UTC at DCE start, for RA_RQST/DEC_RQST.
Known targets receive a *differential* geometric direction correction using
heliocentric spacecraft states plus the builtin ERFA barycentric Sun. This
retains the mission's clock/barycentric anchor, not an independent orbit
solution. The central-pixel subarray offset is prescribed by IRAC Handbook
Appendix D.1.5; full frames retain a nominal FRAMTIME/2 midpoint without a
calibrated pixel/Fowler phase correction. Mission clock uncertainty is typically
20 ms (worst case below 70 ms); rounded headers and orbit errors also remain.
Cached Horizons checks at four epochs show header-state differences up to 57 ms;
this is evidence of a limitation, not a bound for all observations. Target
coordinates are static ICRS directions; no proper-motion/parallax or additional
relativistic propagation solution is applied.

References:
https://irsa.ipac.caltech.edu/data/SPITZER/docs/irac/iracinstrumenthandbook/60/
https://irsa.ipac.caltech.edu/data/SPITZER/docs/irac/iracinstrumenthandbook/32/
https://irsa.ipac.caltech.edu/docs/knowledgebase/spitzer_general.html
"""

import json
from functools import lru_cache
from importlib.metadata import version as package_version
from pathlib import Path

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord, get_body_barycentric_posvel
from astropy.time import Time

SECONDS_PER_DAY = 86400.0
JD_MJD_OFFSET = 2400000.5
C_KM_S = 299792.458
SUBARRAY_MIDPOINT_OFFSET_S = 0.003231
TIMING_VERSION = "irac-midpoint-v3"


def frame_count(header):
    """Validate supported BCD dimensions and return number of exposures."""
    ndim = int(header["NAXIS"])
    if ndim == 2:
        return 1
    if ndim == 3 and int(header["NAXIS3"]) > 0:
        return int(header["NAXIS3"])
    raise ValueError("Expected a 2D IRAC image or 3D subarray cube")


def tdb_minus_utc(header):
    """Mission TDB-UTC seconds; never treat a barycentric time as geocentric."""
    try:
        offset = float(header["ET_OBS"]) - float(header["UTCS_OBS"])
    except KeyError as error:
        raise ValueError("TDB requires both ET_OBS and UTCS_OBS") from error
    if not np.isfinite(offset) or not 60 < offset < 80:
        raise ValueError(f"Implausible IRAC TDB-UTC offset: {offset} seconds")
    return offset


def convert_time(
    values,
    source_format="bmjd",
    source_scale="tdb",
    output_format="bmjd",
    output_scale="tdb",
    header=None,
):
    """Explicit format/scale conversion of already barycentric timestamps.

    UTC -> TDB requires the corresponding BCD clock header (one DCE only).
    This does NOT change the direction, observer, or exposure reference.
    Equal conventions are an identity, so conversion cannot be applied twice.
    """
    for fmt in (source_format, output_format):
        if fmt not in ("bmjd", "bjd"):
            raise ValueError("time_format must be bmjd or bjd")
    if source_scale not in ("utc", "tdb") or output_scale not in ("utc", "tdb"):
        raise ValueError("time_scale must be utc or tdb")
    result = np.asarray(values, dtype=float).copy()
    if source_format == "bjd":
        result -= JD_MJD_OFFSET
    if source_scale != output_scale:
        if header is None:
            raise ValueError("Scale conversion requires the matching BCD header")
        result += (
            tdb_minus_utc(header)
            / SECONDS_PER_DAY
            * (1 if output_scale == "tdb" else -1)
        )
    if output_format == "bjd":
        result += JD_MJD_OFFSET
    return result


@lru_cache(maxsize=32768)
def _sun_state(et):
    epoch = Time(2451545.0, et / SECONDS_PER_DAY, format="jd", scale="tdb")
    pos, vel = get_body_barycentric_posvel("sun", epoch, ephemeris="builtin")
    return pos.xyz.to_value(u.km), vel.xyz.to_value(u.km / u.s)


def target_correction(header, target_coord, offsets):
    """Seconds added to header-reference midpoints, including light-time rate.

    Verify the requested direction against the rounded BMJD/MJD difference
    before using it. A 20 ms closure bound allows header rounding (8.64 ms)
    and the builtin solar ephemeris; it is NOT an absolute timing guarantee.
    Linear propagation over a DCE is limited to 130 s (acceleration neglected,
    about 0.2 ms at 1 AU). Missing/inconsistent states fail, never fall back.
    """
    try:
        state = np.array([header["SPTZR_" + k] for k in ("X", "Y", "Z")], float)
        velocity = np.array([header["SPTZR_V" + k] for k in ("X", "Y", "Z")], float)
        ra, dec = float(header["RA_RQST"]), float(header["DEC_RQST"])
        et = float(header["ET_OBS"])
        anchor = (
            float(header["BMJD_OBS"]) - float(header["MJD_OBS"])
        ) * SECONDS_PER_DAY
    except KeyError as error:
        raise ValueError(f"Target timing requires header {error.args[0]}") from error
    if not np.isfinite(np.r_[state, velocity, ra, dec, et, anchor]).all():
        raise ValueError("Nonfinite spacecraft state or requested direction")
    if not 0 <= ra < 360 or not -90 <= dec <= 90:
        raise ValueError("Invalid RA_RQST/DEC_RQST")
    if not 1e8 < np.linalg.norm(state) < 2e8 or np.linalg.norm(velocity) > 50:
        raise ValueError("Invalid heliocentric spacecraft state (km, km/s required)")
    offsets = np.asarray(offsets)
    if np.any(offsets < 0) or np.any(offsets > 130):
        raise ValueError("DCE exceeds validated 130 s linear-state interval")
    sun, sun_velocity = _sun_state(et)
    pos = state + sun
    reference = SkyCoord(ra, dec, unit="deg", frame="icrs").cartesian.xyz.value
    target_coord = as_target(target_coord)
    target = target_coord.icrs.cartesian.xyz.value
    if target.shape != (3,):
        raise ValueError("A single ICRS target coordinate is required")
    target = target / np.linalg.norm(target)
    closure = np.dot(pos, reference) / C_KM_S - anchor
    if abs(closure) > 0.020:
        raise ValueError(
            f"Header barycentric direction does not close ({closure:.6f} s); "
            "check IER pointing/ephemeris"
        )
    return (
        np.dot(pos, target - reference)
        + offsets * np.dot(velocity + sun_velocity, target)
    ) / C_KM_S


def exposure_midpoints(header, time_scale="tdb", target_coord=None, time_format="bmjd"):
    """Absolute BMJD_TDB midpoints; optional BJD or explicit legacy UTC.

    Subarrays: (i+1/2)*FRAMTIME + 0.003231 s, central pixel (Handbook D.1.5).
    Full frame: nominal FRAMTIME/2, NOT a calibrated pixel/Fowler midpoint.
    Without a target, retain the header direction/correction within the DCE.
    With a target, apply validated differential direction and velocity terms.
    """
    count = frame_count(header)
    start, cadence = float(header["BMJD_OBS"]), float(header["FRAMTIME"])
    if not np.isfinite(start) or not np.isfinite(cadence) or cadence <= 0:
        raise ValueError("BMJD_OBS must be finite and FRAMTIME finite and positive")
    offsets = (np.arange(count, dtype=float) + 0.5) * cadence
    if int(header["NAXIS"]) == 3:
        offsets += SUBARRAY_MIDPOINT_OFFSET_S
    if target_coord is not None:
        offsets = offsets + target_correction(header, target_coord, offsets)
    return convert_time(
        start + offsets / SECONDS_PER_DAY,
        source_scale="utc",
        output_scale=time_scale,
        output_format=time_format,
        header=header,
    )


def timing_metadata(time_scale="tdb", target_coord=None, time_format="bmjd"):
    convert_time(
        [], output_scale=time_scale, source_scale=time_scale, output_format=time_format
    )
    coord = as_target(target_coord) if target_coord is not None else None
    return {
        "dependencies": {
            name: package_version(name) for name in ("astropy", "numpy", "pyerfa")
        },
        "version": TIMING_VERSION,
        "format": time_format.upper(),
        "scale": time_scale.upper(),
        "jd_mjd_offset": JD_MJD_OFFSET,
        "reference": "exposure midpoint",
        "target_icrs_deg": [float(coord.ra.deg), float(coord.dec.deg)]
        if coord
        else None,
        "barycentric": (
            "target_differential_header_state_linear"
            if coord
            else "header_reference_fixed_per_dce"
        ),
        "header_direction": "RA_RQST, DEC_RQST (ICRS degrees)",
        "ephemeris": "header heliocentric J2000 state + ERFA builtin Sun",
        "subarray_offset_seconds": SUBARRAY_MIDPOINT_OFFSET_S,
        "fullframe": "nominal FRAMTIME/2; no pixel/Fowler readout correction",
        "accuracy": (
            "inherits mission clock and header orbit; not precision ephemeris timing"
        ),
        "pointing_caveat": (
            "IER direction must pass geometric closure when target supplied"
        ),
        "binning": "median of timestamps of surviving flux samples",
        "time_std": "population standard deviation in days, not clock error",
    }


def timing_header(time_scale="tdb", target_coord=None, time_format="bmjd", **extra):
    meta = timing_metadata(time_scale, target_coord, time_format)
    meta.update(extra)
    return (
        f"SPCA_TIMING={TIMING_VERSION}; {meta['format']}_{meta['scale']}; midpoint\n"
        "SPCA_TIMING_JSON=" + json.dumps(meta, sort_keys=True)
    )


def read_timing(path):
    """Read declarations, never infer a scale from numerical magnitude."""
    declarations, markers = [], []
    with Path(path).open() as stream:
        for line in stream:
            if line.strip() and not line.lstrip().startswith("#"):
                break
            if "SPCA_TIMING=" in line:
                markers.append(line.split("SPCA_TIMING=", 1)[1].strip())
            if "SPCA_TIMING_JSON=" in line:
                declarations.append(json.loads(line.split("SPCA_TIMING_JSON=", 1)[1]))
    if len(declarations) != 1:
        raise ValueError(
            f"Missing or ambiguous timing metadata: {path}; "
            "legacy products require audited correction"
        )
    meta = declarations[0]
    if (
        meta.get("format") not in ("BMJD", "BJD")
        or meta.get("scale") not in ("UTC", "TDB")
        or not meta.get("version")
        or not meta.get("barycentric")
    ):
        raise ValueError(f"Incomplete timing metadata: {path}")
    if markers and (
        len(markers) != 1
        or markers[0].split(";")[:2]
        != [meta["version"], " " + meta["format"] + "_" + meta["scale"]]
    ):
        raise ValueError(f"Conflicting timing markers: {path}")
    return meta


def model_times(path, values):
    """Normalize declared TDB products to the BMJD_TDB fitting contract."""
    meta = read_timing(path)
    if meta["scale"] != "TDB":
        raise ValueError(
            "Legacy UTC requires matching headers and timing_correction first"
        )
    return convert_time(values, source_format=meta["format"].lower())


def validate_preloaded_time(
    metadata, time_scale="tdb", target_coord=None, time_format="bmjd"
):
    """Require callers supplying arrays to declare their absolute time basis."""
    expected = timing_metadata(time_scale, target_coord, time_format)
    keys = ("version", "format", "scale", "barycentric", "target_icrs_deg")
    if metadata is None or any(metadata.get(k) != expected[k] for k in keys):
        raise ValueError(
            "Preloaded arrays require matching input_time_metadata; "
            "do not relabel legacy or relative times"
        )


def as_target(value):
    """Normalize a scalar SkyCoord or a pair of ICRS degrees."""
    coord = (
        value.icrs
        if isinstance(value, SkyCoord)
        else SkyCoord(*value, unit="deg", frame="icrs")
    )
    if not coord.isscalar or not np.isfinite([coord.ra.deg, coord.dec.deg]).all():
        raise ValueError("Target must be a finite scalar ICRS direction")
    return coord


def load_aor_breaks(full_path, analysis_dir, snippet, aors):
    """Load validated dataset-specific boundaries; never use generic aorBreaks.txt.

    Corrected products store the canonical filename beside the selected full
    table; native reductions store it in analysis/chN. A malformed preferred
    file is an error, not permission to try another file. AOR identities and
    timing declarations must match the selected dataset and photometry.
    """
    from .irac import aor_breaks_name

    full_path = Path(full_path)
    name = aor_breaks_name(snippet)
    selected = full_path.parent / name
    path = selected if selected.exists() else Path(analysis_dir) / name
    if not path.is_file():
        raise ValueError(
            f"Missing dataset-specific AOR boundaries: {path}; "
            "generic aorBreaks.txt is not a safe substitute"
        )
    photometry, metadata = read_timing(full_path), read_timing(path)
    for key in ("version", "format", "scale", "barycentric", "target_icrs_deg"):
        if photometry.get(key) != metadata.get(key):
            raise ValueError(f"AOR boundary timing does not match photometry: {path}")
    starts = metadata.get("aor_starts", {})
    if len(aors) == 0:
        raise ValueError("Expected science AOR identities are required")
    if set(starts) != set(aors):
        raise ValueError(
            f"AOR boundary identities do not match selected dataset: {path}"
        )
    start_values = np.sort(np.asarray(list(starts.values()), dtype=float))
    payload = [
        line
        for line in path.read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    values = np.loadtxt(payload, ndmin=1) if payload else np.array([])
    if (
        not np.isfinite(start_values).all()
        or np.any(np.diff(start_values) <= 0)
        or values.shape != (len(starts) - 1,)
        or not np.allclose(values, start_values[1:], rtol=0, atol=2e-10)
    ):
        raise ValueError(f"Invalid dataset-specific AOR boundaries: {path}")
    return model_times(path, values)
