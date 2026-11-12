"""Timing references, sample associations, and fail-closed product repair."""

import json
from functools import partial

import numpy as np
import pytest
from astropy.io import fits

from SPCA.irac_timing import tdb_minus_utc
from SPCA.Photometry_Common import bin_array, get_time
from SPCA.timing_correction import (
    INVENTORY_FORMAT,
    correct_products,
    make_inventory,
    reconstruct,
)

get_time = partial(get_time, time_scale="utc")

# Independent instrument reference in seconds; do not import the implementation
# constant when checking cadence. IRAC HPP, Timing Offsets Between Frames:
# https://irsa.ipac.caltech.edu/data/SPITZER/docs/irac/irachpp/timing/
REFERENCE_RESET_PHASE_S = 3.231 / 1000


def timing_header(n=64, frame=2.0, start=55000.0, channel=1):
    return dict(
        NAXIS=3 if n > 1 else 2,
        NAXIS1=32 if n > 1 else 256,
        NAXIS2=32 if n > 1 else 256,
        NAXIS3=n,
        BMJD_OBS=start,
        FRAMTIME=frame,
        EXPTIME=frame * 0.8,
        ET_OBS=300000066.184,
        UTCS_OBS=300000000.0,
        CHNLNUM=channel,
    )


@pytest.mark.parametrize("channel", range(1, 5))
@pytest.mark.parametrize("frame", [0.02, 0.1, 0.4, 2.0, 12.0, 1.7])
def test_subarray_reference_intervals_and_endpoints(channel, frame):
    """Verify the IRAC central-pixel cadence and 3.231-ms midpoint offset across
    channels.
    """
    header = timing_header(frame=frame, channel=channel)
    t = get_time(header, np.array([0, 57, 63]))
    # Handbook D.1 independently specifies the midpoint relative to resets.
    offsets = (t - header["BMJD_OBS"]) * 86400
    assert offsets[0] == pytest.approx(frame / 2 + REFERENCE_RESET_PHASE_S, abs=4e-7)
    assert offsets[-1] == pytest.approx(
        63.5 * frame + REFERENCE_RESET_PHASE_S, abs=4e-7
    )
    np.testing.assert_allclose(np.diff(t) * 86400, frame, rtol=0, atol=7e-7)
    assert not np.ma.getmaskarray(t).any()  # rejection must not change row order


@pytest.mark.parametrize("frame,exposure", [(0.6, 0.4), (12.0, 10.4), (50.0, 46.8)])
def test_fullframe_midpoint_uses_fowler_frame_not_integration(frame, exposure):
    """Use FRAMTIME/2 for nominal full-frame timing even when EXPTIME differs."""
    header = timing_header(n=1, frame=frame)
    header["EXPTIME"] = exposure
    assert (get_time(header, [])[0] - header["BMJD_OBS"]) * 86400 == pytest.approx(
        frame / 2, abs=4e-7
    )


def test_nonstandard_cube_and_consecutive_commands_not_extrapolated():
    """Anchor each cube independently so command gaps survive unusual cube lengths."""
    for n in (2, 7, 64):
        h1 = timing_header(n=n, frame=0.4)
        h2 = timing_header(n=n, frame=0.4, start=55000 + (n * 0.4 + 1.8) / 86400)
        times = np.concatenate([get_time(h1), get_time(h2)])
        assert (times[n] - times[n - 1]) * 86400 == pytest.approx(2.2, abs=1e-6)


def test_tdb_header_conversion_real_handbook_units():
    """Convert barycentric UTC using mission-clock seconds and require both clock
    keywords.
    """
    h = timing_header(n=1)
    # Mission ET/UTC clocks, in seconds. Neither is itself the light-time correction.
    assert tdb_minus_utc(h) == pytest.approx(66.184, abs=1e-7)
    difference = (get_time(h, time_scale="tdb") - get_time(h)) * 86400
    np.testing.assert_allclose(difference, 66.184, rtol=0, atol=1e-6)
    del h["ET_OBS"]
    with pytest.raises(ValueError, match="ET_OBS and UTCS_OBS"):
        get_time(h, time_scale="tdb")
    # The compatible UTC path still works without TDB keywords.
    assert np.isfinite(get_time(h)).all()


@pytest.mark.parametrize(
    "key,value",
    [("BMJD_OBS", np.nan), ("FRAMTIME", -1), ("FRAMTIME", np.inf), ("NAXIS", 4)],
)
def test_invalid_timing_fails(key, value):
    """Reject nonfinite anchors, invalid frame durations and unsupported timing
    geometry.
    """
    h = timing_header()
    h[key] = value
    with pytest.raises(ValueError):
        get_time(h)


@pytest.mark.parametrize("ignore", [[-1], [64], [0.5]])
def test_invalid_frame_positions_fail(ignore):
    """Reject fractional or out-of-cube frame exclusions instead of changing row
    identity.
    """
    with pytest.raises(ValueError, match="integer cube positions"):
        get_time(timing_header(), ignore)


def test_masked_and_partial_bin_midpoints_are_measured_from_survivors():
    """Compute median times from surviving rows, retaining the final partial bin."""
    times = get_time(timing_header(n=7))
    times.mask[[0, 1, 4, 5]] = True
    med, std = bin_array(times, 3)
    np.testing.assert_allclose(med, times.data[[2, 3, 6]], rtol=0, atol=1e-11)
    np.testing.assert_array_equal(std, 0)


def legacy_products(tmp_path, n=64, mask_policy="finite_flux", time_scale="utc"):
    entries = [
        dict(id="r2/later.fits", aor="r2", header=timing_header(n=n, start=55000.01)),
        dict(id="r1/first.fits", aor="r1", header=timing_header(n=n)),
    ]
    inventory = dict(
        format=INVENTORY_FORMAT,
        channel="ch1",
        bcd_count=2,
        aor_counts={"r1": 1, "r2": 1},
        entries=entries,
    )
    headers = tmp_path / "headers.json"
    headers.write_text(json.dumps(inventory))
    old, new = reconstruct(inventory, time_scale)
    data = np.zeros((len(old), 8))
    data[:, 0] = 1000 + np.arange(len(old))
    data[:, 1] = old
    data[:, 2:] = np.arange(len(old))[:, None]
    data[1:3, 0] = np.nan
    data[min(4, len(data) - 1), 3] = np.nan  # independent centroid mask must survive
    size = 16 if n == 1 else n
    bins = np.zeros(((len(data) + size - 1) // size, 16))
    for i, start in enumerate(range(0, len(data), size)):
        part = data[start : start + size]
        good = np.isfinite(part[:, 0])
        times = part[good, 1] if mask_policy == "finite_flux" else part[:, 1]
        bins[i, :4] = [
            np.nanmedian(part[:, 0]),
            np.nanstd(part[:, 0]),
            np.median(times),
            np.std(times),
        ]
    full = tmp_path / "full.dat"
    binned = tmp_path / "binned.dat"
    np.savetxt(full, data, header="Flux, Time, x, y, wx, wy, bg, npp")
    np.savetxt(binned, bins, header="Flux, Flux std, Time, Time std, ...")
    return dict(
        headers=headers, full=full, binned=binned, bin_size=size, time_scale=time_scale
    ), new


@pytest.mark.parametrize("mask_policy", ["finite_flux", "all_rows"])
@pytest.mark.parametrize("time_scale", ["utc", "tdb"])
def test_repair_validates_masks_rebins_and_preserves_non_time_bytes(
    tmp_path, mask_policy, time_scale
):
    """Validate both historical mask policies and preserve every non-time token in
    corrected copies.
    """
    args, new = legacy_products(
        tmp_path, mask_policy=mask_policy, time_scale=time_scale
    )
    originals = {p: p.read_bytes() for p in tmp_path.iterdir()}
    report = correct_products(**args)
    assert not report["applied"]
    assert report["binned_mask_policy"] == mask_policy
    assert set(tmp_path.iterdir()) == set(originals)
    args.update(
        output_full=tmp_path / "fixed.dat",
        output_binned=tmp_path / "bins-fixed.dat",
        audit=tmp_path / "audit.json",
    )
    result = correct_products(**args, apply=True)
    assert result["applied"]
    np.testing.assert_array_equal(np.loadtxt(args["output_full"])[:, 1], new)
    for source, dest, time_cols in [
        (args["full"], args["output_full"], {1}),
        (args["binned"], args["output_binned"], {2, 3}),
    ]:
        before = [
            line.split()
            for line in source.read_text().splitlines()
            if line and not line.startswith("#")
        ]
        after = [
            line.split()
            for line in dest.read_text().splitlines()
            if line and not line.startswith("#")
        ]
        for row_before, row_after in zip(before, after):
            for i in range(len(row_before)):
                if i not in time_cols:
                    assert row_before[i] == row_after[i]
        assert len(before) == len(after)
    for path, payload in originals.items():
        assert path.read_bytes() == payload
    with pytest.raises(FileExistsError):
        correct_products(**args, apply=True)
    args["full"] = args["output_full"]
    with pytest.raises(ValueError, match="Already versioned"):
        correct_products(**args)


def test_repair_fullframe_utc_unchanged_and_tdb_shift(tmp_path):
    """Keep nominal full-frame UTC midpoints and apply the independently specified
    TDB offset.
    """
    args, _ = legacy_products(tmp_path, n=1)
    report = correct_products(**args)
    assert report["full_shift_seconds"]["max"] == 0
    args["time_scale"] = "tdb"
    report = correct_products(**args)
    assert report["full_shift_seconds"]["median"] == pytest.approx(66.184, abs=1e-6)


@pytest.mark.parametrize(
    "corruption",
    [
        "time",
        "count",
        "bin_time",
        "bin_std",
        "bin_flux",
        "unknown_bin_size",
        "duplicate_header",
    ],
)
def test_repair_refuses_unverifiable_inputs_without_writes(tmp_path, corruption):
    """Refuse mismatched rows, bin statistics, fluxes or header identities before
    writing.
    """
    args, _ = legacy_products(tmp_path)
    if corruption == "unknown_bin_size":
        args["bin_size"] = 13
    elif corruption == "duplicate_header":
        inventory = json.loads(args["headers"].read_text())
        inventory["entries"][1] = inventory["entries"][0]
        args["headers"].write_text(json.dumps(inventory))
    else:
        file = args["full"] if corruption in ("time", "count") else args["binned"]
        data = np.loadtxt(file)
        if corruption == "count":
            data = data[:-1]
        else:
            col = {"time": 1, "bin_time": 2, "bin_std": 3, "bin_flux": 0}[corruption]
            data[0, col] += 1
        np.savetxt(file, data)
    before = set(tmp_path.iterdir())
    with pytest.raises(ValueError):
        correct_products(
            **args,
            output_full=tmp_path / "out.dat",
            output_binned=tmp_path / "out-bin.dat",
            audit=tmp_path / "audit.json",
            apply=True,
        )
    assert set(tmp_path.iterdir()) == before


def test_repair_rejects_same_paths_and_cannot_run_without_audit(tmp_path):
    """Require separate output paths and an audit before applying any correction."""
    args, _ = legacy_products(tmp_path)
    with pytest.raises(ValueError, match="distinct paths"):
        correct_products(**args, output_full=args["full"])
    with pytest.raises(ValueError, match="explicit output paths"):
        correct_products(**args, apply=True)


def test_inventory_explicit_aors_counts_and_header_only_reconstruction(tmp_path):
    """Require explicit AOR/count selection and reconstruct timing solely from
    inventoried headers.
    """
    root = tmp_path / "data/ch3"
    folder = root / "r123/ch3/bcd"
    folder.mkdir(parents=True)
    header = fits.Header(timing_header(n=1, channel=3))
    header["AORKEY"] = 123
    # Structural FITS keywords must describe the real array, not be set manually.
    for key in list(header):
        if key.startswith("NAXIS"):
            del header[key]
    fits.PrimaryHDU(np.ones((256, 256)), header).writeto(folder / "first_bcd.fits")
    inventory = make_inventory(root, "ch3", ["r123"], 1)
    assert inventory["bcd_count"] == 1
    assert len(reconstruct(inventory, "tdb")[0]) == 1
    with pytest.raises(ValueError, match="Expected 2 BCDs"):
        make_inventory(root, "ch3", ["r123"], 2)
    with pytest.raises(ValueError, match="No BCDs"):
        make_inventory(root, "ch3", ["r124"], 1)


def test_header_inventory_paths_are_never_dereferenced(tmp_path):
    """Treat inventory filenames as identities rather than instructions to read
    external data.
    """
    args, _ = legacy_products(tmp_path)
    inv = json.loads(args["headers"].read_text())
    inv["entries"][0]["id"] = "/unavailable/header-only/never-open-this.fits"
    args["headers"].write_text(json.dumps(inv))
    assert correct_products(**args)["rows"] == 128


def test_correction_exports_consistent_aor_breaks(tmp_path):
    """Export AOR boundaries in the same corrected time convention as the light
    curve.
    """
    args, new = legacy_products(tmp_path, time_scale="tdb")
    args.update(
        output_full=tmp_path / "fixed.dat",
        output_binned=tmp_path / "bin-fixed.dat",
        output_breaks=tmp_path / "breaks.txt",
        audit=tmp_path / "audit.json",
    )
    report = correct_products(**args, apply=True)
    assert report["aor_breaks_bmjd"] == [new[64]]
    np.testing.assert_allclose(
        np.loadtxt(args["output_breaks"]), new[64], rtol=0, atol=1e-11
    )


def test_repair_empty_bins_and_partial_final_bin(tmp_path):
    """Validate empty and partial legacy bins without changing their row membership."""
    args, _ = legacy_products(tmp_path, n=7)
    full = np.loadtxt(args["full"])
    full[:7, 0] = np.nan
    np.savetxt(args["full"], full)
    size = 5
    rows = []
    for first in range(0, len(full), size):
        part = full[first : first + size]
        part = part[np.isfinite(part[:, 0])]
        row = np.zeros(16)
        row[:4] = (
            [
                np.median(part[:, 0]),
                np.std(part[:, 0]),
                np.median(part[:, 1]),
                np.std(part[:, 1]),
            ]
            if len(part)
            else [np.nan] * 4
        )
        rows.append(row)
    np.savetxt(args["binned"], rows)
    args["bin_size"] = size
    report = correct_products(**args)
    assert report["rejected_flux_rows"] == 7
    assert report["binned_mask_policy"] == "finite_flux"


def test_legacy_companion_timing_uses_shared_absolute_reference():
    """Append absolute companion midpoints using the shared IRAC timing calculation."""
    from SPCA.irac_timing import exposure_midpoints
    from SPCA.Photometry_Companion import get_time as companion_time

    h = timing_header()
    h["AINTBEG"], h["ATIMEEND"] = 123.0, 251.0
    hdus = fits.HDUList([fits.PrimaryHDU(header=fits.Header())])
    hdus[0].header.update(h)
    values = [54999.0]
    assert companion_time(hdus, values, time_scale="tdb") is values
    assert values[0] == 54999.0
    np.testing.assert_array_equal(values[1:], exposure_midpoints(h, "tdb"))
