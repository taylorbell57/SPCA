"""Contributor-aware sampling times without changes to photometry or row bins."""

import numpy as np
import pytest
from test_irac_timing import legacy_products

from SPCA.irac_timing import read_timing
from SPCA.Photometry_Common import bin_array, bin_timestamps
from SPCA.Photometry_Companion import binning_data
from SPCA.timing_correction import correct_products


@pytest.mark.parametrize(
    "rejected,expected",
    [
        ([], 5),
        ([0, 1], 7),
        ([1, 2], 7),
        ([3, 4], 3),
        ([2], 5),
        ([0, 2, 4], 5),
        ([2, 3, 4], 2),
        ([0, 1, 2, 3, 4], np.nan),
    ],
)
@pytest.mark.parametrize("rejection", ["mask", "nan"])
def test_time_uses_actual_flux_contributors(rejected, expected, rejection):
    """Asymmetric flux rejection must shift the median sampling time.

    With only the first two of five exposures surviving, both median flux and
    representative time use those two rows: the time is 2 s, rather than 5 s.
    Masked underlying values must never reappear in either statistic.
    """
    time = np.array([1.0, 3.0, 5.0, 7.0, 9.0])
    flux = np.ma.array([100.0, 200.0, 300.0, 400.0, 500.0], mask=False)
    if rejection == "mask":
        flux.mask[rejected] = True
    else:
        flux[rejected] = np.nan
    original = flux.copy()
    median, scatter = bin_timestamps(time, flux, 5)
    np.testing.assert_allclose(median, [expected], equal_nan=True)
    good = np.setdiff1d(np.arange(5), rejected)
    expected_std = np.std(time[good]) if len(good) else np.nan
    np.testing.assert_allclose(scatter, [expected_std], equal_nan=True)
    np.testing.assert_array_equal(flux.data, original.data)
    np.testing.assert_array_equal(flux.mask, original.mask)


@pytest.mark.parametrize("missing", [np.nan, np.inf, -np.inf])
def test_missing_contributing_time_invalidates_time_only(missing):
    """A valid flux lacking time makes the bin time unknowable, preserving flux."""
    times = [1, missing, 9]
    flux = [2, 4, 6]
    assert np.isnan(bin_timestamps(times, flux, 3)).all()
    assert bin_array(flux, 3)[0][0] == 4
    np.testing.assert_allclose(bin_timestamps(times, [2, np.nan, 6], 3), [[5], [4]])


def test_masked_timestamp_is_not_restored_by_flux_mask():
    """A masked finite time remains unavailable even when its flux is valid."""
    times = np.ma.array([1, 3, 9], mask=[0, 1, 0])
    assert np.isnan(bin_timestamps(times, [1, 1, 1], 3)).all()


def test_uneven_sampling_partial_bins_and_aor_gap():
    """Keep consecutive row membership across AOR gaps and retain the final bin.

    Uneven sampling distinguishes the chosen median time (3 s) from the mean
    (34 s). A bin spanning two AORs remains one bin for legacy compatibility.
    """
    time = [1, 3, 98, 100, 102, 1000, 1002]
    median, scatter = bin_timestamps(time, np.ones(7), 3)
    np.testing.assert_array_equal(median, [3, 102, 1002])
    np.testing.assert_allclose(
        scatter, [np.std([1, 3, 98]), np.std([100, 102, 1000]), 0]
    )
    np.testing.assert_array_equal(binning_data(np.arange(7), 3)[0], [1, 4, 6])


def test_pld_time_requires_common_pixel_contributors():
    """Pixel medians with different exposure masks have no single sampling time."""
    stamps = np.ma.array(np.ones((6, 2)), mask=False)
    stamps.mask[:2, :] = True
    stamps.mask[3, 0] = True
    median, scatter = bin_timestamps([1, 3, 9, 11, 15, 17], stamps, 3)
    np.testing.assert_allclose(median, [9, np.nan], equal_nan=True)
    np.testing.assert_allclose(scatter, [0, np.nan], equal_nan=True)


@pytest.mark.parametrize("mask_policy", ["finite_flux", "all_rows"])
def test_legacy_validation_and_corrected_contributors_are_separate(
    tmp_path, mask_policy
):
    """Validate the saved historical times before computing contributor times.

    Source time medians/scatter keep their historical expectations. Only the
    output times/scatter follow finite full-resolution fluxes, with provenance
    distinguishing the two policies and all non-time tokens preserved.
    """
    args, new = legacy_products(
        tmp_path, n=7, mask_policy=mask_policy, time_scale="tdb"
    )
    source = np.loadtxt(args["binned"])
    args.update(
        output_full=tmp_path / "fixed.dat",
        output_binned=tmp_path / "fixed-bin.dat",
        audit=tmp_path / "audit.json",
    )
    report = correct_products(**args, apply=True)
    full = np.loadtxt(args["full"])
    expected = []
    for first in (0, 7):
        good = np.isfinite(full[first : first + 7, 0])
        times = new[first : first + 7][good]
        expected.append([np.median(times), np.std(times)])
    output = np.loadtxt(args["output_binned"])
    np.testing.assert_allclose(output[:, 2:4], expected, atol=1e-11, rtol=0)
    np.testing.assert_array_equal(
        np.delete(output, [2, 3], axis=1), np.delete(source, [2, 3], axis=1)
    )
    meta = read_timing(args["output_binned"])
    assert report["binned_mask_policy"] == meta["historical_bin_mask"] == mask_policy
    assert meta["bin_mask"] == "finite_flux"
    if mask_policy == "all_rows":
        assert output[0, 2] != source[0, 2]


@pytest.mark.parametrize("method", ["psf", "pld"])
def test_extraction_uses_postprocessing_contributors(tmp_path, monkeypatch, method):
    """Exercise saved PSF/PLD bins with asymmetric rejections and missing times.

    Hold clipping fixed to isolate which rows feed the existing median flux
    estimator. A partially observed first bin must use 2 s, a missing time in
    the second bin must invalidate timing alone, and the final row survives.
    """
    from SPCA import Photometry_PLD, Photometry_PSF
    from SPCA.irac_timing import timing_metadata

    module = Photometry_PSF if method == "psf" else Photometry_PLD
    count = 11
    flux = np.ma.array(np.ones(count), mask=False)
    flux.mask[2:5] = True
    time = 55000 + np.arange(1, 22, 2) / 86400
    time[7] = np.nan
    monkeypatch.setattr(module, "clip_data", lambda values, *a, **kw: values)
    if method == "psf":
        monkeypatch.setattr(
            module,
            "fit_2DGaussian",
            lambda *a, **kw: (flux, *[np.ones(count) for _ in range(4)]),
        )
        extra = {}
    else:
        stamps = np.ma.array(np.ones((count, 9)), mask=False)
        stamps.mask[2:5, :] = True
        monkeypatch.setattr(module, "get_pixel_values", lambda *a, **kw: stamps)
        extra = {"stamp_sizes": [3]}
    module.get_lightcurve(
        str(tmp_path),
        "r123",
        "ch1",
        "sample",
        image_stack_input=np.ones((count, 32, 32)),
        bg=np.ones(count),
        bg_err=np.ones(count),
        time=time,
        input_time_metadata=timing_metadata(),
        bin_size=5,
        savePlots=False,
        ncpu=1,
        **extra,
    )
    path = next(tmp_path.rglob("*_binned_*.dat"))
    table = np.loadtxt(path)
    column = 2 if method == "psf" else 18
    np.testing.assert_allclose(
        table[:, column],
        [55000 + 2 / 86400, np.nan, time[-1]],
        rtol=0,
        atol=1e-11,
        equal_nan=True,
    )
    np.testing.assert_allclose(
        table[:, column + 1], [1 / 86400, np.nan, 0], rtol=0, atol=1e-11, equal_nan=True
    )
    # The missing time never removes a pixel/flux sample from its original bin.
    np.testing.assert_array_equal(table[:, 0], np.ones(3))


def test_legacy_empty_bin_and_cross_aor_membership_are_preserved(tmp_path):
    """Repair all-row timing without splitting bins that straddle two AORs.

    The first bin has finite historical time but no flux contributors, so its
    corrected time/scatter become NaN. The second bin spans the AOR boundary;
    its saved flux and row membership must remain exactly as supplied.
    """
    args, new = legacy_products(tmp_path, n=7, time_scale="tdb")
    full = np.loadtxt(args["full"])
    full[:5, 0] = np.nan
    np.savetxt(args["full"], full)
    args["bin_size"] = 5
    rows = []
    for first in range(0, len(full), 5):
        part = full[first : first + 5]
        flux = part[np.isfinite(part[:, 0]), 0]
        row = np.zeros(16)
        row[:4] = [
            np.median(flux) if len(flux) else np.nan,
            np.std(flux) if len(flux) else np.nan,
            np.median(part[:, 1]),
            np.std(part[:, 1]),
        ]
        rows.append(row)
    np.savetxt(args["binned"], rows)
    args.update(
        output_full=tmp_path / "fixed.dat",
        output_binned=tmp_path / "fixed-bin.dat",
        audit=tmp_path / "audit.json",
    )
    report = correct_products(**args, apply=True)
    output = np.loadtxt(args["output_binned"])
    assert report["binned_mask_policy"] == "all_rows"
    assert np.isnan(output[0, 2:4]).all()
    np.testing.assert_allclose(
        output[1, 2:4], [np.median(new[5:10]), np.std(new[5:10])], rtol=0, atol=1e-11
    )
    np.testing.assert_array_equal(
        np.delete(output, [2, 3], axis=1), np.delete(rows, [2, 3], axis=1)
    )
