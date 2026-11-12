"""Explicit scale/format, real direction correction, and product compatibility."""

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from astropy import units as u
from astropy.coordinates import SkyCoord, get_body_barycentric
from astropy.io import fits
from astropy.time import Time
from test_irac_timing import legacy_products

from SPCA.irac_timing import (
    JD_MJD_OFFSET,
    convert_time,
    exposure_midpoints,
    model_times,
    read_timing,
    target_correction,
    timing_header,
    timing_metadata,
    validate_preloaded_time,
)
from SPCA.Photometry_Common import get_time
from SPCA.Photometry_Companion import get_time as companion_time
from SPCA.timing_correction import correct_products

HEADERS = json.loads(
    (Path(__file__).parent / "fixtures/irac_real_headers.json").read_text()
)
TARGET = SkyCoord(
    HEADERS[0]["header"]["RA_RQST"] + 0.02,
    HEADERS[0]["header"]["DEC_RQST"] - 0.01,
    unit="deg",
)


def test_default_bmjd_tdb_and_explicit_bjd():
    """Default to BMJD_TDB and change JD/MJD format without applying a second scale
    correction.
    """
    h = HEADERS[0]["header"]
    utc = exposure_midpoints(h, "utc")
    tdb = get_time(h)
    np.testing.assert_allclose(
        (tdb - utc) * 86400, h["ET_OBS"] - h["UTCS_OBS"], rtol=0, atol=1e-6
    )
    bjd = get_time(h, time_format="bjd")
    np.testing.assert_array_equal(bjd, tdb + JD_MJD_OFFSET)
    np.testing.assert_array_equal(convert_time(tdb), tdb)  # no double scale conversion
    np.testing.assert_allclose(
        convert_time(bjd, source_format="bjd"), tdb, rtol=0, atol=2.4e-10
    )  # JD float quantization
    np.testing.assert_array_equal(convert_time(utc, source_scale="utc", header=h), tdb)
    with pytest.raises(ValueError, match="matching BCD"):
        convert_time(utc, source_scale="utc")
    with pytest.raises(ValueError):
        convert_time(tdb, source_format="instrument")


@pytest.mark.parametrize(
    "entry", HEADERS, ids=lambda e: e["channel"] + "-" + e["id"].split("/")[-1]
)
def test_real_header_directions_and_target_geometry(entry):
    """Check real-header direction delays against an independent barycentric vector
    calculation.
    """
    h = entry["header"]
    target = SkyCoord(h["RA_RQST"] + 0.02, h["DEC_RQST"] - 0.01, unit="deg")
    # Independent vector computation, retaining the rounded header anchor.
    epoch = Time(2451545.0, h["ET_OBS"] / 86400, format="jd", scale="tdb")
    pos = np.array([h["SPTZR_" + a] for a in "XYZ"])
    pos += get_body_barycentric("sun", epoch, ephemeris="builtin").xyz.to_value(u.km)
    ref = SkyCoord(h["RA_RQST"], h["DEC_RQST"], unit="deg").cartesian.xyz.value
    direction = np.dot(pos, target.cartesian.xyz.value - ref) / 299792.458
    closure = np.dot(pos, ref) / 299792.458 - (h["BMJD_OBS"] - h["MJD_OBS"]) * 86400
    assert abs(closure) < 0.00865  # independent bound from two 7-decimal-day headers
    assert target_correction(h, target, [0])[0] == pytest.approx(direction, abs=1e-12)
    t = get_time(h, target_coord=target)
    assert np.isfinite(t).all() and np.all(np.diff(t) > 0)
    if h["NAXIS"] == 3:
        # Light travel time evolves within real 2-second cubes, ~12 ms by the end.
        corr = (t - get_time(h)) * 86400
        assert 0 < corr[-1] - corr[0] < 0.013


def test_target_and_header_reference_are_distinct_and_fail_closed():
    """Distinguish requested and target sight lines and reject incomplete or
    inconsistent states.
    """
    h = HEADERS[0]["header"].copy()
    assert timing_metadata()["barycentric"] == "header_reference_fixed_per_dce"
    assert timing_metadata(target_coord=TARGET)["barycentric"].startswith(
        "target_differential"
    )
    assert abs(target_correction(h, TARGET, [0])[0]) > 1e-4
    h["RA_RQST"] += 1
    with pytest.raises(ValueError, match="does not close"):
        get_time(h, target_coord=TARGET)
    del h["SPTZR_X"]
    with pytest.raises(ValueError, match="SPTZR_X"):
        get_time(h, target_coord=TARGET)
    assert np.isfinite(get_time(h)).all()  # explicit header-reference capability
    with pytest.raises(ValueError, match="130 s"):
        target_correction(HEADERS[0]["header"], TARGET, [131])


def test_companion_absolute_default_and_tuple_target():
    """Use absolute TDB companion timing and accept equivalent SkyCoord and degree-
    pair targets.
    """
    header = fits.Header(HEADERS[0]["header"])
    hdus = fits.HDUList([fits.PrimaryHDU(header=header)])
    # PrimaryHDU with no data resets NAXIS; restore timing geometry explicitly.
    hdus[0].header.update(header)
    result = companion_time(hdus, [], target_coord=TARGET)
    np.testing.assert_array_equal(result, get_time(header, target_coord=TARGET))
    np.testing.assert_array_equal(
        result, get_time(header, target_coord=(TARGET.ra.deg, TARGET.dec.deg))
    )


def test_saved_metadata_and_model_normalization(tmp_path):
    """Normalize declared BJD_TDB and refuse missing, conflicting or UTC model-input
    metadata.
    """
    path = tmp_path / "phot.dat"
    np.savetxt(path, [[1, 2455000.5]], header=timing_header(time_format="bjd"))
    assert model_times(path, [2455000.5])[0] == 55000
    path.write_text("# UTC or TDB?\n1 55000\n")
    with pytest.raises(ValueError, match="Missing or ambiguous"):
        model_times(path, [55000])
    np.savetxt(path, [[1, 55000]], header=timing_header("utc"))
    with pytest.raises(ValueError, match="Legacy UTC"):
        model_times(path, [55000])
    path.write_text(
        "# " + timing_header().replace("\n", "\n# ") + "\n"
        "# SPCA_TIMING_JSON=" + json.dumps(timing_metadata()) + "\n"
    )
    with pytest.raises(ValueError, match="ambiguous"):
        read_timing(path)
    with pytest.raises(ValueError, match="Preloaded"):
        validate_preloaded_time(None)
    with pytest.raises(ValueError, match="Preloaded"):
        validate_preloaded_time(timing_metadata("utc"))
    validate_preloaded_time(timing_metadata())


def test_correction_bjd_format_and_double_conversion(tmp_path):
    """Write audited BJD_TDB copies and refuse to correct an already versioned
    product again.
    """
    args, new = legacy_products(tmp_path, time_scale="tdb")
    args.update(
        output_full=tmp_path / "fixed.dat",
        output_binned=tmp_path / "fixed-bin.dat",
        audit=tmp_path / "audit.json",
    )
    report = correct_products(**args, time_format="bjd", apply=True)
    out = np.loadtxt(args["output_full"])
    np.testing.assert_allclose(out[:, 1] - JD_MJD_OFFSET, new, rtol=0, atol=2.4e-10)
    assert report["full_shift_seconds"]["max"] < 80
    assert read_timing(args["output_full"])["format"] == "BJD"
    args["full"] = args["output_full"]
    with pytest.raises(ValueError, match="Already versioned"):
        correct_products(**args)


def test_custom_ephemeris_format_is_explicit():
    """Require explicit BJD input format and a TDB scale for custom transit epochs."""
    from SPCA import Decorrelation_helper as dh

    args = dict(
        rootpath="",
        planet="test",
        channel="ch1",
        rp=0.1,
        a=8.0,
        per=1.0,
        inc=88.0,
        e=0.0,
        argp=0.0,
        Tstar=6000.0,
        logg=4.0,
        feh=0.0,
    )
    with patch.object(dh, "getTstarBright", return_value=(6000.0, 100.0)):
        a = dh.loadCustomData(t0=55000.0, **args)
        b = dh.loadCustomData(t0=2455000.5, t0_format="bjd", **args)
        assert a["t0"] == b["t0"] == 55000.0
        assert a["time_convention"] == "BMJD_TDB"
        with pytest.raises(ValueError, match="epoch must be TDB"):
            dh.loadCustomData(t0=55000.0, t0_scale="utc", **args)


def test_conflicting_markers_refused(tmp_path):
    """Reject contradictory text and JSON timing declarations."""
    path = tmp_path / "bad.dat"
    text = timing_header()
    text = text.replace("irac-midpoint-v3; BMJD_TDB", "irac-midpoint-v3; BMJD_UTC")
    np.savetxt(path, [[1.0, 55000.0]], header=text)
    with pytest.raises(ValueError, match="Conflicting"):
        read_timing(path)


@pytest.mark.parametrize("method", ["psf", "pld"])
def test_other_photometry_saves_declared_tdb(tmp_path, method):
    """Save PSF and PLD time columns with consistent TDB metadata and all-valid bin
    medians.
    """
    from SPCA import Photometry_PLD, Photometry_PSF

    module = Photometry_PSF if method == "psf" else Photometry_PLD
    y, x = np.mgrid[:32, :32]
    image = 100 * np.exp(-((x - 15.2) ** 2 + (y - 15.1) ** 2) / 2.0)
    frames = np.ma.array(np.repeat(image[None], 24, axis=0))
    time = 55000 + np.arange(24) / 86400
    (tmp_path / "planet/analysis/ch1").mkdir(parents=True)
    kwargs = dict(
        ncpu=1,
        image_stack_input=frames,
        bg=np.ones(24),
        bg_err=np.ones(24),
        time=time,
        input_time_metadata=timing_metadata(),
        savePlots=False,
        highpassWidth=6,
        bin_size=4,
    )
    if method == "pld":
        kwargs["stamp_sizes"] = [3]
    module.get_lightcurve(str(tmp_path), "r123", "ch1", "planet", **kwargs)
    full = next(tmp_path.rglob("*_full_*.dat"))
    bins = next(tmp_path.rglob("*_binned_*.dat"))
    full_column, bin_column = (1, 2) if method == "psf" else (9, 18)
    np.testing.assert_array_equal(np.loadtxt(full)[:, full_column], time)
    np.testing.assert_allclose(
        np.loadtxt(bins)[:, bin_column],
        np.median(time.reshape(-1, 4), axis=1),
        rtol=0,
        atol=1e-11,
    )
    assert read_timing(full) == read_timing(bins)
    assert read_timing(full)["scale"] == "TDB"


def test_companion_full_model_loader_uses_declared_time_column(tmp_path):
    """Read companion full-resolution time from its declared third column."""
    from SPCA.helpers import get_full_data

    time = 55000 + np.arange(16) / 86400
    table = np.column_stack(
        [np.ones(16), np.full(16, 0.01), time, np.full((16, 6), 15.0)]
    )
    path = tmp_path / "companion.dat"
    np.savetxt(
        path, table, header=timing_header(product_kind="companion", time_column=2)
    )
    result = get_full_data(str(tmp_path) + "/", path.name, "poly2", nFrames=1)
    np.testing.assert_array_equal(result[1], time)


@pytest.mark.parametrize("kind", ["utc", "old_version", "other_target", "compatible"])
def test_preloaded_resume_requires_matching_timing(kind):
    """Require preloaded timestamps to match version, scale and science-target
    metadata.
    """
    metadata = timing_metadata(target_coord=TARGET)
    if kind == "utc":
        metadata = timing_metadata("utc", TARGET)
    elif kind == "old_version":
        metadata["version"] = "obsolete"
    elif kind == "other_target":
        metadata["target_icrs_deg"] = [0.0, 0.0]
    if kind == "compatible":
        validate_preloaded_time(metadata, target_coord=TARGET)
    else:
        with pytest.raises(ValueError, match="Preloaded"):
            validate_preloaded_time(metadata, target_coord=TARGET)


def test_mission_clock_against_independent_astropy_scale_conversion():
    """Check mission-clock TDB against Astropy within the header clocks' millisecond
    rounding.
    """
    for entry in HEADERS:
        h = entry["header"]
        expected = Time(h["DATE_OBS"], scale="utc").tdb
        mission = Time(2451545.0, h["ET_OBS"] / 86400, format="jd", scale="tdb")
        # Both mission clocks are rounded to milliseconds in these header samples.
        assert abs((mission - expected).to_value(u.s)) < 0.002


@pytest.mark.parametrize("coordinates", [(0.0, 0.0), (120.0, -20.0), (250.0, 60.0)])
def test_target_direction_against_cached_horizons(coordinates):
    """Compare target delays with cached Horizons states while accounting for header-
    orbit error.
    """
    from astropy.table import Table

    table = Table.read(Path(__file__).parent / "fixtures/horizons_spitzer.ecsv")
    indexed = {e["id"]: e["header"] for e in HEADERS}
    target = SkyCoord(*coordinates, unit="deg")
    for row in table:
        h = indexed[row["header_id"]]
        requested = SkyCoord(h["RA_RQST"], h["DEC_RQST"], unit="deg")
        difference = target.cartesian.xyz.value - requested.cartesian.xyz.value
        position = np.array([row[a] for a in "xyz"]) * u.au.to(u.km)
        epoch = Time(row["datetime_jd"], format="jd", scale="tdb")
        header_position = np.array([h["SPTZR_" + a] for a in "XYZ"])
        header_position += get_body_barycentric(
            "sun", epoch, ephemeris="builtin"
        ).xyz.to_value(u.km)
        expected = position @ difference / 299792.458
        # Do not mistake header orbit error for precision target-direction timing.
        orbit_bound = (
            np.linalg.norm(header_position - position)
            * np.linalg.norm(difference)
            / 299792.458
        )
        assert (
            abs(target_correction(h, target, [0])[0] - expected) <= orbit_bound + 1e-9
        )


@pytest.mark.parametrize(
    "kind", ["legacy", "utc", "different_target", "different_rows", "compatible"]
)
def test_combined_model_inputs_require_compatible_times(tmp_path, kind):
    """Reject mixed PLD/aperture inputs with incompatible declarations or different
    time rows.
    """
    from SPCA.helpers import require_matching_times

    first, second = tmp_path / "first.dat", tmp_path / "second.dat"
    np.savetxt(
        first,
        [[1.0, 55000.0], [1.0, 55001.0]],
        header=timing_header(target_coord=TARGET),
    )
    header = timing_header(target_coord=TARGET)
    values = [55000.0, 55001.0]
    if kind == "legacy":
        header = "Flux Time"
    elif kind == "utc":
        header = timing_header("utc", TARGET)
    elif kind == "different_target":
        header = timing_header()
    elif kind == "different_rows":
        values[1] += 1.0
    np.savetxt(second, np.column_stack([np.ones(2), values]), header=header)
    if kind == "compatible":
        require_matching_times(first, 1, second, 1)
    else:
        with pytest.raises(ValueError):
            require_matching_times(first, 1, second, 1)
