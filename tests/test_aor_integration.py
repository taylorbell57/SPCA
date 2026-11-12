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
