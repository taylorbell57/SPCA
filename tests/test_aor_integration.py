"""Dataset-specific AOR boundaries and aperture selection across observing epochs."""

from pathlib import Path

import numpy as np
import pytest
from conftest import SYNTHETIC_TARGET as TARGET
from conftest import make_bcd

from SPCA import Decorrelation_helper as dh
from SPCA.irac import aor_breaks_name, aor_tag
from SPCA.irac_timing import load_aor_breaks, timing_header
from SPCA.Photometry_Common import prepare_images


@pytest.mark.parametrize("snippet", ["r101", "101"])
def test_canonical_tag(snippet):
    """Normalize leading-r AOR selectors to dataset-specific filename tags."""
    assert aor_tag(snippet) == "101"
    assert aor_breaks_name(snippet) == "aorBreaks_AORs101.txt"


@pytest.fixture
def native_datasets(tmp_path):
    program = "PID12345_2008"
    analysis = tmp_path / program / "analysis/ch2"
    folder = analysis / "addedBlank/noIgnore/4umExactCircular2_0"
    folder.mkdir(parents=True)
    expected = {}
    for snippet, indices in [("r101", (0, 1)), ("r202", (20, 21))]:
        for suffix, index in zip(("001", "002"), indices):
            make_bcd(tmp_path, "ch2", aor=snippet + suffix, index=index)
        _, _, _, times = prepare_images(
            str(tmp_path) + "/", program, "ch2", snippet, target_coord=TARGET, ncpu=1
        )
        full = folder / f"ch2_datacube_full_AORs{aor_tag(snippet)}.dat"
        np.savetxt(
            full,
            np.column_stack([np.ones(2), times]),
            header=timing_header(target_coord=TARGET),
        )
        expected[snippet] = np.array([times[1]])
    (analysis / "bestPhOption.txt").write_text(str(folder) + "\nignoreFrames = \n100\n")
    return tmp_path, program, analysis, folder, expected


def test_same_program_epochs_ignore_overwritten_generic(native_datasets):
    """Load dataset-specific AOR boundaries even when another epoch overwrote the
    generic file.
    """
    root, program, analysis, _, expected = native_datasets
    # prepare_images has already overwritten the generic file with the second epoch.
    np.testing.assert_allclose(
        np.loadtxt(analysis / "aorBreaks.txt", ndmin=1),
        expected["r202"],
        rtol=0,
        atol=1e-11,
    )
    for snippet in expected:
        (root / program / "analysis/aorSnippet.txt").write_text(snippet)
        result = dh.findPhotometry(str(root) + "/", program, "ch2", "poly2")
        assert result[1] == f"ch2_datacube_binned_AORs{aor_tag(snippet)}.dat"
        assert result[5] == aor_tag(snippet)
        np.testing.assert_allclose(result[7], expected[snippet], rtol=0, atol=1e-11)
    # A valid, current-TDB generic file still cannot replace a missing specific one.
    (analysis / aor_breaks_name("r101")).unlink()
    (root / program / "analysis/aorSnippet.txt").write_text("r101")
    with pytest.raises(
        ValueError, match="generic aorBreaks.txt is not a safe substitute"
    ):
        dh.findPhotometry(str(root) + "/", program, "ch2", "poly3")
    assert not (Path(result[0]) / "poly3").exists()


def test_preferred_boundaries_must_match_dataset_and_scale(native_datasets):
    """Refuse invalid preferred boundary metadata rather than falling back to another
    file.
    """
    _, _, analysis, folder, _ = native_datasets
    full = folder / "ch2_datacube_full_AORs101.dat"
    selected = folder / aor_breaks_name("r101")
    # The path alone is insufficient: reject another epoch's identities.
    selected.write_bytes((analysis / aor_breaks_name("r202")).read_bytes())
    with pytest.raises(ValueError, match="identities"):
        load_aor_breaks(full, analysis, "r101", ["r101001", "r101002"])
    # Do not fall back to the valid analysis-level file after a preferred-file error.
    selected.write_text(selected.read_text().replace("BMJD_TDB", "BMJD_UTC"))
    with pytest.raises(ValueError):
        load_aor_breaks(full, analysis, "r101", ["r101001", "r101002"])


def test_aperture_selection_stays_with_requested_dataset(native_datasets):
    """Select the lowest-scatter aperture only among products belonging to the
    requested epoch.
    """
    root, program, analysis, folder, expected = native_datasets
    other = folder.with_name("4umExactCircular2_5")
    other.mkdir()
    second = folder / "ch2_datacube_full_AORs202.dat"
    second.rename(other / second.name)
    with (analysis / "bestPhOption.txt").open("a") as stream:
        stream.write(str(other) + "\nignoreFrames = \n1\n")
    for snippet, selected in [("r101", folder), ("r202", other)]:
        (root / program / "analysis/aorSnippet.txt").write_text(snippet)
        result = dh.findPhotometry(str(root) + "/", program, "ch2", "poly2")
        assert Path(result[0]) == selected
        np.testing.assert_allclose(result[7], expected[snippet], rtol=0, atol=1e-11)


def test_aperture_selection_with_both_naming_conventions(native_datasets):
    """Rank old/new manifest paths within the requested observing epoch."""
    root, program, analysis, old, _ = native_datasets
    new = old.with_name("ch2ExactCircular2_0")
    new.mkdir()
    name = "ch2_datacube_full_AORs101.dat"
    (new / name).write_bytes((old / name).read_bytes())
    with (analysis / "bestPhOption.txt").open("a") as stream:
        stream.write(str(new) + "\nignoreFrames = \n1\n")
    for snippet, selected in [("r101", new), ("r202", old)]:
        (root / program / "analysis/aorSnippet.txt").write_text(snippet)
        found = dh.findPhotometry(str(root) + "/", program, "ch2", "poly2")
        assert Path(found[0]) == selected


@pytest.mark.parametrize("prefix", ["4um", "ch2"])
@pytest.mark.parametrize("method", ["PSFX", "PLD1_3x3"])
def test_automatic_psf_pld_lookup_keeps_legacy_products(
    native_datasets, prefix, method
):
    """Find old/new names for the selected dataset and extraction settings."""
    root, program, analysis, aperture, _ = native_datasets
    suffix = "PSF" if method == "PSFX" else "PLD_3x3"
    folder = aperture.parent / (prefix + suffix)
    folder.mkdir()
    filename = "ch2_datacube_full_AORs101.dat"
    (folder / filename).write_bytes((aperture / filename).read_bytes())
    (root / program / "analysis/aorSnippet.txt").write_text("r101")
    result = dh.findPhotometry(
        str(root) + "/", program, "ch2", method, pldIgnoreFrames=False
    )
    assert Path(result[0]) == folder


def test_coexisting_reductions_cannot_replace_aor_time_basis(native_datasets):
    """Preserve dataset sidecars when a new reduction requests another time format."""
    root, program, analysis, _, _ = native_datasets
    before = {path: path.read_bytes() for path in analysis.rglob("*") if path.is_file()}
    with pytest.raises(FileExistsError, match="AOR timing already exists"):
        prepare_images(
            str(root) + "/",
            program,
            "ch2",
            "r101",
            target_coord=TARGET,
            time_format="bjd",
            ncpu=1,
        )
    assert all(path.read_bytes() == payload for path, payload in before.items())
    selected = analysis / aor_breaks_name("r101")
    payload = selected.read_bytes()
    prepare_images(str(root) + "/", program, "ch2", "r101", target_coord=TARGET, ncpu=1)
    assert selected.read_bytes() == payload


@pytest.mark.parametrize("time_format", ["bmjd", "bjd"])
@pytest.mark.parametrize("corruption", ["offset", "missing", "extra", "nonfinite"])
def test_sidecar_reuse_checks_numbers_as_well_as_metadata(
    tmp_path, time_format, corruption
):
    """Reject corrupted numerical boundaries even when declarations match exactly.

    The 1-ms offset is much smaller than a relative JD tolerance would admit,
    but exceeds the absolute 17-us serialization allowance. Refusal preserves
    both the dataset sidecar and the shared compatibility file byte-for-byte.
    """
    program = "PID12345_2008"
    for aor, index in [("r101001", 0), ("r101002", 1)]:
        make_bcd(tmp_path, "ch2", aor=aor, index=index)
    args = dict(
        basepath=str(tmp_path) + "/",
        planet=program,
        channel="ch2",
        AOR_snip="r101",
        target_coord=TARGET,
        time_format=time_format,
        ncpu=1,
    )
    prepare_images(**args)
    sidecar = tmp_path / program / "analysis/ch2" / aor_breaks_name("r101")
    lines = sidecar.read_text().splitlines(keepends=True)
    original_metadata = "".join(line for line in lines if line.startswith("#"))
    boundary = np.loadtxt(sidecar, ndmin=1)[0]
    payload = {
        "offset": f"{boundary + 0.001 / 86400:.15f}\n",
        "missing": "\n",
        "extra": f"{boundary:.15f} {boundary + 1:.15f}\n",
        "nonfinite": "nan\n",
    }[corruption]
    sidecar.write_text(original_metadata + payload)
    before = {
        path: path.read_bytes() for path in sidecar.parent.rglob("*") if path.is_file()
    }
    with pytest.raises(ValueError, match="Invalid dataset-specific AOR boundaries"):
        prepare_images(**args)
    assert all(path.read_bytes() == content for path, content in before.items())


@pytest.mark.parametrize("indicator", ["marked_export", "relocated_table"])
def test_preparation_allows_corrected_sibling_dataset(native_datasets, indicator):
    """Reduce epoch 202 while preserving corrected epoch 101 in the same channel."""
    from SPCA.timing_correction import CORRECTED_TREE_MARKER

    root, program, analysis, folder, _ = native_datasets
    if indicator == "marked_export":
        export = analysis / "epoch101/corrected"
        export.mkdir(parents=True)
        (export / CORRECTED_TREE_MARKER).write_text("{}")
    else:
        path = folder / "ch2_datacube_full_AORs101.dat"
        np.savetxt(
            path,
            np.loadtxt(path),
            header=timing_header(target_coord=TARGET, product_role="timing_corrected"),
        )
    before = {path: path.read_bytes() for path in analysis.rglob("*") if path.is_file()}
    _, _, _, time = prepare_images(
        str(root) + "/", program, "ch2", "r202", target_coord=TARGET, ncpu=1
    )
    assert len(time) == 2
    assert all(path.read_bytes() == payload for path, payload in before.items())


def test_preparation_protects_exact_relocated_boundary_output(native_datasets):
    """A corrected boundary file is immutable even outside a marked export tree."""
    root, program, analysis, _, _ = native_datasets
    path = analysis / "aorBreaks.txt"
    np.savetxt(path, [55000], header=timing_header(product_role="timing_corrected"))
    before = {p: p.read_bytes() for p in analysis.rglob("*") if p.is_file()}
    with pytest.raises(ValueError, match="protected"):
        prepare_images(
            str(root) + "/", program, "ch2", "r101", target_coord=TARGET, ncpu=1
        )
    assert all(p.read_bytes() == payload for p, payload in before.items())


def test_compatible_corrected_sidecar_can_be_reused_read_only(native_datasets):
    """Reuse a relocated corrected sidecar without writing to its protected tree."""
    import json

    from SPCA.irac_timing import read_timing
    from SPCA.timing_correction import CORRECTED_TREE_MARKER

    root, program, analysis, _, _ = native_datasets
    selected = analysis / aor_breaks_name("r101")
    metadata = read_timing(selected)
    metadata["product_role"] = "timing_corrected"
    export = analysis / "corrected101"
    export.mkdir()
    (export / CORRECTED_TREE_MARKER).write_text("{}")
    relocated = export / selected.name
    # Metadata and data remain mutually consistent; only the product role changes.
    declaration = "# SPCA_TIMING_JSON=" + json.dumps(metadata) + "\n"
    values = np.loadtxt(selected, ndmin=1)
    relocated.write_text(declaration + " ".join(f"{v:.15f}" for v in values) + "\n")
    selected.unlink()
    selected.symlink_to(relocated)
    before = relocated.read_bytes()
    prepare_images(str(root) + "/", program, "ch2", "r101", target_coord=TARGET, ncpu=1)
    assert relocated.read_bytes() == before
    assert selected.is_symlink()
