"""Synthetic IRAC channel routing, extraction, geometry and frame diagnostics."""

import hashlib
from unittest.mock import patch

import numpy as np
import pytest
from astropy.io import fits
from astropy.wcs import WCS
from conftest import SYNTHETIC_TARGET as TARGET
from conftest import make_bcd

from SPCA import Photometry_Aperture as aperture
from SPCA import frameDiagnosticsBackend as diagnostics
from SPCA.irac import CHANNELS, aperture_prefix, validate_channel
from SPCA.irac_timing import read_timing, timing_metadata
from SPCA.Photometry_Common import get_fnames, prepare_image


@pytest.mark.parametrize("channel,prefix", zip(CHANNELS, CHANNELS))
def test_channel_naming(channel, prefix):
    """Use unambiguous channel prefixes for new photometry in all four IRAC channels."""
    assert aperture_prefix(channel) == prefix


@pytest.mark.parametrize("invalid", ["ch0", "ch5", "3", 3, "CH3", "../ch2"])
def test_unknown_channel_never_falls_back(invalid):
    """Reject unknown channel spellings instead of silently selecting another
    detector.
    """
    with pytest.raises(ValueError):
        validate_channel(invalid)


@pytest.mark.parametrize("channel", CHANNELS)
def test_fullframe_extraction_best_only_and_metadata(tmp_path, channel):
    """Save only the winning aperture pair with channel-specific calibration and
    declared TDB.
    """
    files = [make_bcd(tmp_path, channel, index=i) for i in range(17)]
    hashes = [hashlib.sha256(path.read_bytes()).hexdigest() for path in files]
    aperture.get_lightcurve(
        str(tmp_path),
        "r123",
        channel,
        "PID12345_2008",
        ncpu=1,
        save=True,
        onlyBest=True,
        savePlots=False,
        showPlots=False,
        target_coord=TARGET,
        r=[2.0, 2.5],
        moveCentroids=[True, False],
        bin_size=16,
        highpassWidth=64,
        time_scale="tdb",
        centroid_bounds=(
            (11, 21, 11, 21) if channel in ("ch3", "ch4") else (13, 18, 13, 18)
        ),
        background_bounds=(
            (8, 24, 8, 24) if channel in ("ch3", "ch4") else (11, 19, 11, 19)
        ),
    )
    products = list((tmp_path / f"PID12345_2008/analysis/{channel}").rglob("*.dat"))
    assert len(products) == 2  # four candidates evaluated, only one saved
    full_path = next(path for path in products if "_full_" in path.name)
    binned_path = next(path for path in products if "_binned_" in path.name)
    assert full_path.parent.name.startswith(aperture_prefix(channel))
    assert read_timing(full_path) == read_timing(binned_path)
    full = np.loadtxt(full_path)
    assert np.isfinite(full).all()
    # Independent analytic source placement in the WCS crop (fixed at first BCD).
    # The legacy 5x5 box truncates broad PSFs; the Si:As 10x10 box
    # should recover this synthetic source to substantially better precision.
    centroid_atol = 0.02 if channel in ("ch3", "ch4") else 0.20
    np.testing.assert_allclose(full[:, 2], 15.3, atol=centroid_atol)
    np.testing.assert_allclose(full[:, 3], 15.7, atol=centroid_atol)
    np.testing.assert_allclose(
        (full[:, 1] - 54773) * 86400, np.arange(17) * 12 + 6 + 65.182, rtol=0, atol=1e-6
    )
    assert read_timing(full_path)["scale"] == "TDB"
    assert read_timing(full_path)["format"] == "BMJD"
    assert np.loadtxt(binned_path).shape == (2, 16)
    assert hashes == [hashlib.sha256(path.read_bytes()).hexdigest() for path in files]


@pytest.mark.parametrize("channel", CHANNELS)
def test_subarray_discovery_header_mismatch_and_diagnostics(tmp_path, channel):
    """Preserve subarray exclusions, enforce header channel identity and propagate
    diagnostic errors.
    """
    path = make_bcd(tmp_path, channel, subarray=True)
    filenames, lengths = get_fnames(
        str(tmp_path / f"PID12345_2008/data/{channel}"), "r123"
    )
    assert filenames == [str(path)] and lengths == [1]
    data = prepare_image(
        str(tmp_path),
        "r123",
        filenames,
        lengths,
        ignoreFrames=[0, 57],
        channel=channel,
        time_scale="tdb",
    )
    assert data.shape == (64, 1025)
    assert data.mask[[0, 57], :-1].all()
    assert not data.mask[:, -1].any()
    np.testing.assert_allclose(np.diff(data[:, -1]) * 86400, 2, atol=7e-7, rtol=0)
    wrong = "ch1" if channel != "ch1" else "ch2"
    with pytest.raises(ValueError, match="CHNLNUM"):
        prepare_image(str(tmp_path), "r123", filenames, lengths, channel=wrong)
    with patch.object(
        diagnostics, "get_lightcurve", side_effect=RuntimeError("extract")
    ) as call:
        with pytest.raises(RuntimeError, match="extract"):
            diagnostics.run_diagnostics(
                "PID12345_2008", channel, "r123", str(tmp_path), False, savePlot=False
            )
        call.assert_called_once()


def test_subarray_star_mask_respects_xy_and_stamp_edges(tmp_path):
    """Mask nearby stars with correct x/y order and clip masks at stamp edges."""
    path = make_bcd(tmp_path, "ch3", subarray=True)
    w = WCS(fits.getheader(path), naxis=[1, 2])
    coords = [w.pixel_to_world(3, 8), w.pixel_to_world(0, 0), w.pixel_to_world(45, 45)]
    stars = [(c.ra, c.dec) for c in coords]
    prepared = prepare_image(str(tmp_path), "r", [str(path)], [1], maskStars=stars)
    mask = prepared[:, :-1].reshape(64, 32, 32).mask
    assert mask[:, 7:10, 2:5].all()
    assert not mask[:, 2:5, 7:10].any()
    assert mask[:, :2, :2].all()
    assert not mask[:, -2:, -2:].any()


def test_multiple_aors_sort_cube_measurements_with_masks(tmp_path):
    """Sort AOR exposures and masks together while preserving independent timing
    anchors.
    """
    from SPCA.Photometry_Common import prepare_images

    files = [
        make_bcd(tmp_path, "ch4", aor=aor, index=i, subarray=True)
        for aor, i in [("r123", 2), ("r456", 0), ("r789", 4)]
    ]
    frames, bg, _, time = prepare_images(
        str(tmp_path) + "/",
        "PID12345_2008",
        "ch4",
        "r",
        ignoreFrames=[1, 63],
        ncpu=1,
        time_scale="tdb",
    )
    for block, file in enumerate([files[1], files[0], files[2]]):
        header = fits.getheader(file)
        good = np.ones(64, bool)
        good[[1, 63]] = False
        selection = slice(block * 64, (block + 1) * 64)
        original = fits.getdata(file)
        factor = header["GAIN"] * header["EXPTIME"] / header["FLUXCONV"]
        np.testing.assert_allclose(
            (frames[selection] + bg[selection, None, None])[good],
            (original * factor)[good],
        )
        assert frames[selection].mask[[1, 63]].all()
        assert not time[selection].mask.any()
        assert (time[selection][0] - header["BMJD_OBS"]) * 86400 == pytest.approx(
            # FRAMTIME/2 + IRAC reset phase + mission TDB-UTC seconds.
            1.0 + 3.231e-3 + 65.182,
            abs=1e-6,
        )
    assert np.all(np.diff(time) > 0)


def test_diagnostics_reject_interleaved_cubes(tmp_path):
    """Reject overlapping cubes whose rows cannot support frame-position diagnostics."""
    first = make_bcd(tmp_path, "ch3", index=0, subarray=True)
    second = make_bcd(tmp_path, "ch3", index=1, subarray=True)
    with fits.open(second, mode="update") as hdus:
        hdus[0].header["BMJD_OBS"] = fits.getheader(first)["BMJD_OBS"] + 1 / 86400
    with patch.object(diagnostics, "get_lightcurve") as extract:
        with pytest.raises(ValueError, match="Overlapping cubes"):
            diagnostics.run_diagnostics(
                "PID12345_2008", "ch3", "r", str(tmp_path), False
            )
    extract.assert_not_called()


def test_onlybest_invalid_data_never_saves_full_grid(tmp_path):
    """Refuse to save an entire aperture grid when no finite RMS ranking exists."""
    frames = np.ma.masked_all((8, 32, 32))
    with pytest.raises(ValueError, match="refusing to save an entire"):
        aperture.get_lightcurve(
            str(tmp_path),
            "r123",
            "ch3",
            "sample",
            image_stack_input=frames,
            input_time_metadata=timing_metadata(),
            bg=np.ones(8),
            bg_err=np.ones(8),
            time=np.arange(8) + 55000,
            r=[2.0, 3.0],
            moveCentroids=[True, False],
            onlyBest=True,
            bin_size=2,
            highpassWidth=4,
            save=True,
            savePlots=False,
            ncpu=1,
        )
    assert not list(tmp_path.rglob("*.dat"))


@pytest.mark.parametrize("channel,legacy", [("ch1", "3um"), ("ch2", "4um")])
@pytest.mark.parametrize(
    "suffix", ["ExactCircular2_3_movingCentroid", "PSF", "PLD_3x3"]
)
def test_legacy_lookup_and_coexisting_dataset_copies(tmp_path, channel, legacy, suffix):
    """Read legacy datasets and refuse ambiguous copies across conventions."""
    from SPCA.irac import resolve_photometry_dir

    filename = f"{channel}_datacube_full_AORs123.dat"
    old = tmp_path / (legacy + suffix)
    new = tmp_path / (channel + suffix)
    old.mkdir()
    (old / filename).write_text("legacy science bytes")
    assert resolve_photometry_dir(tmp_path, channel, suffix, filename) == str(old) + "/"
    new.mkdir()
    # Another dataset under the new prefix must not hide the requested legacy one.
    (new / filename.replace("123", "456")).write_text("other epoch")
    assert resolve_photometry_dir(tmp_path, channel, suffix, filename) == str(old) + "/"
    (new / filename).write_text("new science bytes")
    with pytest.raises(ValueError, match="Ambiguous photometry"):
        resolve_photometry_dir(tmp_path, channel, suffix, filename)
    assert (old / filename).read_text() == "legacy science bytes"
    assert (new / filename).read_text() == "new science bytes"


@pytest.mark.parametrize("channel", ["ch3", "ch4"])
def test_long_wavelength_channels_have_no_legacy_aliases(tmp_path, channel):
    """Avoid introducing wavelength aliases for newly supported Si:As channels."""
    from SPCA.irac import photometry_prefixes, resolve_photometry_dir

    assert photometry_prefixes(channel) == (channel,)
    with pytest.raises(FileNotFoundError):
        resolve_photometry_dir(tmp_path, channel, "PSF", "missing.dat")


def test_output_creation_protects_existing_and_relocated_products(tmp_path):
    """Protect exact outputs while permitting new datasets beside corrected files."""
    from SPCA.irac_timing import timing_header
    from SPCA.Photometry_Common import create_folder

    path = tmp_path / "ch1ExactCircular2_0"
    create_folder(path, products=("full.dat",))
    (path / "full.dat").write_bytes(b"original science")
    with pytest.raises(FileExistsError, match="Photometry already exists"):
        create_folder(path, True, True, products=("full.dat",))
    assert (path / "full.dat").read_bytes() == b"original science"
    # A corrected file remains protected even when exported without root markers.
    np.savetxt(
        path / "relocated.dat",
        [[1, 55000]],
        header=timing_header(product_role="timing_corrected"),
    )
    before = (path / "relocated.dat").read_bytes()
    assert create_folder(path, True, True, products=("different_epoch.dat",)) == path
    assert (path / "relocated.dat").read_bytes() == before
    with pytest.raises(ValueError, match="protected"):
        create_folder(path, True, True, products=("relocated.dat",))


@pytest.mark.parametrize("channel", ["ch1", "ch2"])
def test_phoenix_approximation_requires_complete_finite_window(channel):
    """Reject truncated/invalid stellar grids before evaluating brightness sums."""
    from SPCA.Decorrelation_helper import _phoenix_window

    wavelength = np.linspace(0.05, 5.499975, 1000)
    flux = np.ones_like(wavelength)
    selected = _phoenix_window(wavelength, flux, channel)
    assert selected.any()
    for wave, values in [
        (wavelength[:650], flux[:650]),
        (wavelength[::-1], flux),
        (wavelength, flux[:-1]),
    ]:
        with pytest.raises(ValueError, match="coverage"):
            _phoenix_window(wave, values, channel)
    flux[selected] = np.nan
    with pytest.raises(ValueError, match="finite PHOENIX"):
        _phoenix_window(wavelength, flux, channel)


@pytest.mark.parametrize("channel", ["ch3", "ch4"])
def test_phoenix_long_wavelength_limit_is_explicit(channel):
    """Explain the selected files' wavelength limit before any stellar download."""
    from SPCA.Decorrelation_helper import getTstarBright

    with pytest.raises(NotImplementedError, match="5.5 um"):
        getTstarBright("", "", channel, {})


def test_table_notebook_keeps_each_discovered_mode_parent(tmp_path):
    """Keep distinct PLD results when old/new directories contain the same mode."""
    import json
    from pathlib import Path

    parent = tmp_path / "sample/analysis/ch2/addedBlank/noIgnore"
    aperture_dir = parent / "4umExactCircular2_1"
    for folder, mode in [
        (aperture_dir, "Poly2"),
        (parent / "4umPLD_3x3", "PLD1_3x3"),
        (parent / "ch2PLD_3x3", "PLD1_3x3"),
    ]:
        (folder / mode).mkdir(parents=True)
    (tmp_path / "sample/analysis/ch2/bestPhOption.txt").write_text(str(aperture_dir))
    notebook = json.loads((Path(__file__).parents[1] / "MakeTables.ipynb").read_text())
    source = "".join(notebook["cells"][3]["source"])
    # Execute discovery only; table construction requires unrelated fit products.
    discovery = source.split("    # find the best modes")[0]
    namespace = {
        "mainpath": str(tmp_path) + "/",
        "planets": ["sample"],
        "channels": ["ch2"],
    }
    exec(compile(discovery, "MakeTables.ipynb", "exec"), namespace)
    entries = namespace["mode_entries"]
    assert len(entries) == 3
    assert set(entries) == {
        ("Poly2", str(aperture_dir) + "/"),
        ("PLD1_3x3", str(parent / "4umPLD_3x3") + "/"),
        ("PLD1_3x3", str(parent / "ch2PLD_3x3") + "/"),
    }
