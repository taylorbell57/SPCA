"""Independent datasets may share output directories without replacing science."""

from pathlib import Path

import numpy as np
import pytest

from SPCA import Decorrelation_helper as dh
from SPCA import Photometry_Aperture, Photometry_PLD, Photometry_PSF, helpers
from SPCA.irac import resolve_photometry_dir
from SPCA.irac_timing import timing_header, timing_metadata
from SPCA.timing_correction import CORRECTED_TREE_MARKER, assert_reduction_tree

MODULES = {
    "aperture": Photometry_Aperture,
    "psf": Photometry_PSF,
    "pld": Photometry_PLD,
}


@pytest.fixture
def extract(tmp_path):
    """Run native-pixel extraction on a small, smoothly varying synthetic star."""
    n = 24
    y, x = np.mgrid[:32, :32]
    star = np.exp(-((x - 15.2) ** 2 + (y - 15.1) ** 2) / 2)
    frames = (1000 + 2 * np.sin(np.arange(n)))[:, None, None] * star

    def run(method, tag="101", channel="ch1", save=True):
        module = MODULES[method]
        if save:
            (tmp_path / "sample/analysis" / channel).mkdir(parents=True, exist_ok=True)
        extra = {}
        if method == "aperture":
            extra = dict(r=[2.4], edges=["Exact"], moveCentroids=[False])
        elif method == "pld":
            extra = dict(stamp_sizes=[3])
        module.get_lightcurve(
            str(tmp_path),
            "r" + tag,
            channel,
            "sample",
            image_stack_input=np.ma.array(frames.copy()),
            bg=np.ones(n),
            bg_err=np.ones(n),
            time=55000 + int(tag) + np.arange(n) / 86400,
            input_time_metadata=timing_metadata(),
            save=save,
            savePlots=False,
            showPlots=False,
            ncpu=1,
            bin_size=4,
            highpassWidth=8,
            **extra,
        )
        if save:
            suffix = {"aperture": "ExactCircular2_4", "psf": "PSF", "pld": "PLD_3x3"}[
                method
            ]
            return (
                tmp_path
                / "sample/analysis"
                / channel
                / "addedBlank/noIgnore"
                / (channel + suffix)
                / f"{channel}_datacube_full_AORs{tag}.dat"
            )

    return run


def snapshot(root):
    """Capture file contents so preservation checks include partial products."""
    return {path: path.read_bytes() for path in root.rglob("*") if path.is_file()}


@pytest.mark.parametrize("method", MODULES)
@pytest.mark.parametrize("corrected", [False, True])
def test_new_dataset_reuses_technique_without_changing_existing_science(
    tmp_path, extract, method, corrected
):
    """Add a second dataset beside native or relocated corrected full/bin pairs."""
    first = extract(method)
    folder = first.parent
    if corrected:
        for path in folder.glob("*.dat"):
            np.savetxt(
                path,
                np.loadtxt(path),
                header=timing_header(product_role="timing_corrected"),
            )
    # A separate marked export under the same channel must not block extraction.
    marked = tmp_path / "sample/analysis/ch1/another_dataset/corrected"
    marked.mkdir(parents=True)
    (marked / CORRECTED_TREE_MARKER).write_text("{}")
    before = snapshot(folder)
    second = extract(method, "202")
    assert second.parent == folder
    assert all(path.read_bytes() == payload for path, payload in before.items())
    assert second.exists()
    assert second.with_name(second.name.replace("_full_", "_binned_")).exists()
    with pytest.raises(FileExistsError, match="Photometry already exists"):
        extract(method, "202")
    if corrected:
        with pytest.raises(ValueError, match="protected"):
            extract(method)
        # Recursive deletion remains forbidden even though new files can be added.
        with pytest.raises(ValueError, match="protected"):
            assert_reduction_tree(folder, recursive=True)
    assert all(path.read_bytes() == payload for path, payload in before.items())


@pytest.mark.parametrize("method", MODULES)
@pytest.mark.parametrize("interruption", ["partial_full", "before_binned"])
def test_interrupted_output_pair_is_never_implicitly_resumed(
    tmp_path, extract, monkeypatch, method, interruption
):
    """Leave interrupted files intact and refuse retries until the user moves them.

    An error during the full table can leave a truncated file; an error before
    the binned table leaves a complete full table. Neither state authorizes
    overwriting or manufacturing the missing mate on the next run.
    """
    original = np.savetxt

    def interrupted(path, *args, **kwargs):
        if interruption == "partial_full" and "_full_" in str(path):
            Path(path).write_text("# interrupted full product\n")
            raise OSError("simulated interrupted write")
        if interruption == "before_binned" and "_binned_" in str(path):
            raise OSError("simulated interrupted write")
        return original(path, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(np, "savetxt", interrupted)
        with pytest.raises(OSError, match="simulated interrupted write"):
            extract(method)
    before = snapshot(tmp_path)
    assert len(list(tmp_path.rglob("*_full_*.dat"))) == 1
    assert not list(tmp_path.rglob("*_binned_*.dat"))
    with pytest.raises(FileExistsError, match="Photometry already exists"):
        extract(method)
    assert snapshot(tmp_path) == before


@pytest.mark.parametrize("method", MODULES)
def test_binned_only_pair_is_preserved_on_retry(tmp_path, extract, method):
    """A surviving binned table also blocks regeneration of either output file."""
    full = extract(method)
    full.unlink()
    before = snapshot(tmp_path)
    with pytest.raises(FileExistsError, match="Photometry already exists"):
        extract(method)
    assert snapshot(tmp_path) == before


@pytest.mark.parametrize("method", MODULES)
def test_read_only_arrays_are_allowed_in_marked_channel(tmp_path, extract, method):
    """Preloaded, unsaved extractions perform no writes in a protected tree."""
    channel = tmp_path / "sample/analysis/ch1"
    channel.mkdir(parents=True)
    (channel / CORRECTED_TREE_MARKER).write_text("{}")
    before = snapshot(tmp_path)
    extract(method, save=False)
    assert snapshot(tmp_path) == before


@pytest.mark.parametrize("channel,legacy", [("ch1", "3um"), ("ch2", "4um")])
@pytest.mark.parametrize("convention", ["legacy", "channel"])
@pytest.mark.parametrize("binned", [False, True])
def test_actual_model_inputs_load_both_conventions_and_psfx_centroids(
    tmp_path, extract, channel, legacy, convention, binned
):
    """Load real extractor tables through aperture, PLD and PSFX model loaders.

    PSFX must use the selected PSF table's centroids/widths with identical time
    rows. Exercise absolute resolved and relative sibling PSF paths and require
    explicit selection when old/new directories contain duplicate datasets.
    """
    paths = {method: extract(method, channel=channel) for method in MODULES}
    if convention == "legacy":
        for method, path in paths.items():
            destination = path.parent.with_name(legacy + path.parent.name[3:])
            path.parent.rename(destination)
            paths[method] = destination / path.name
    paths = {
        method: (
            path.with_name(path.name.replace("_full_", "_binned_")) if binned else path
        )
        for method, path in paths.items()
    }
    filename = paths["aperture"].name
    aperture_dir = str(paths["aperture"].parent) + "/"
    psf_dir = resolve_photometry_dir(
        paths["psf"].parent.parent, channel, "PSF", filename
    )
    pld_dir = resolve_photometry_dir(
        paths["pld"].parent.parent, channel, "PLD_3x3", filename
    )
    # Supply the local AOR discovery inputs used by findPhotometry. The
    # extraction itself used explicitly declared preloaded timestamps above.
    from astropy.io import fits

    raw = tmp_path / "sample/data" / channel / "r101" / channel / "bcd"
    raw.mkdir(parents=True)
    fits.PrimaryHDU(header=fits.Header({"BMJD_OBS": 55101.0})).writeto(
        raw / "source.fits"
    )
    analysis = tmp_path / "sample/analysis" / channel
    (analysis.parent / "aorSnippet.txt").write_text("r101")
    (analysis / "bestPhOption.txt").write_text(
        aperture_dir.rstrip("/") + "\nIgnoreFrames = \n1\n"
    )
    np.savetxt(
        analysis / "aorBreaks_AORs101.txt",
        [],
        header=timing_header(aor_starts={"r101": 55101.0}),
    )
    found = dh.findPhotometry(str(tmp_path) + "/", "sample", channel, "PSFX")
    assert found[0] == psf_dir
    found_pld = dh.findPhotometry(
        str(tmp_path) + "/", "sample", channel, "PLD1_3x3", pldIgnoreFrames=False
    )
    assert found_pld[0] == pld_dir
    loader = helpers.get_data if binned else helpers.get_full_data
    extra = {} if binned else {"nFrames": 1}
    ordinary = loader(aperture_dir, filename, "BLISS", **extra)
    psf_table = np.loadtxt(paths["psf"])
    for psf_path in (psf_dir, paths["psf"].parent.name + "/"):
        result = loader(
            aperture_dir, filename, "BLISS_PSFX", foldername_psf=psf_path, **extra
        )
        np.testing.assert_array_equal(result[0], ordinary[0])
        np.testing.assert_array_equal(result[1], ordinary[1])
        columns = (4, 6, 8, 10) if binned else (2, 3, 4, 5)
        for actual, column in zip(result[2:], columns):
            np.testing.assert_array_equal(actual, psf_table[:, column])
    stamps, flux, time = loader(pld_dir, filename, "PLD1_3x3", **extra)
    assert stamps.shape == (9, len(time))
    np.testing.assert_allclose(stamps.sum(axis=0), 1, atol=1e-15, rtol=0)
    np.testing.assert_array_equal(time, ordinary[1])
    for method, suffix in [("psf", "PSF"), ("pld", "PLD_3x3")]:
        source = paths[method]
        other_prefix = channel if convention == "legacy" else legacy
        other = source.parent.with_name(other_prefix + suffix)
        other.mkdir()
        (other / filename).write_bytes(source.read_bytes())
        with pytest.raises(ValueError, match="Ambiguous photometry"):
            resolve_photometry_dir(other.parent, channel, suffix, filename)


def test_interactive_folder_choice_rechecks_product_protection(tmp_path, monkeypatch):
    """Changing a folder interactively cannot bypass relocated-file protection."""
    from SPCA.Photometry_Common import create_folder

    original = tmp_path / "original"
    original.mkdir()
    (original / "other_dataset.dat").write_text("existing science")
    corrected = tmp_path / "corrected"
    corrected.mkdir()
    np.savetxt(
        corrected / "new.dat",
        [[1, 55000]],
        header=timing_header(product_role="timing_corrected"),
    )
    before = snapshot(tmp_path)
    responses = iter(["n", "corrected"])
    monkeypatch.setattr("builtins.input", lambda: next(responses))
    with pytest.raises(ValueError, match="protected"):
        create_folder(str(original), products=("new.dat",))
    assert snapshot(tmp_path) == before
