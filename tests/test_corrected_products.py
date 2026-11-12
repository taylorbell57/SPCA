"""Individual-product correction, provenance and generic tree protection."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from conftest import SYNTHETIC_TARGET, add_synthetic_state
from test_irac_timing import legacy_products

from SPCA.irac import aor_breaks_name
from SPCA.irac_timing import load_aor_breaks, read_timing, timing_header
from SPCA.timing_correction import (
    CORRECTED_TREE_MARKER,
    assert_reduction_tree,
    correct_products,
)


def correction_inputs(tmp_path, channel="ch1", time_format="bmjd"):
    source = tmp_path / "source"
    source.mkdir()
    args, _ = legacy_products(source, time_scale="tdb")
    inventory = json.loads(args["headers"].read_text())
    inventory["channel"] = channel
    for entry in inventory["entries"]:
        entry["header"]["CHNLNUM"] = int(channel[-1])
        add_synthetic_state(entry["header"], SYNTHETIC_TARGET)
    args["headers"].write_text(json.dumps(inventory))
    out = tmp_path / "corrected"
    args.update(
        output_full=out / f"{channel}_datacube_full_AORs.dat",
        output_binned=out / f"{channel}_datacube_binned_AORs.dat",
        output_breaks=out / aor_breaks_name("r"),
        audit=out / "audit.json",
        target_coord=SYNTHETIC_TARGET,
        time_format=time_format,
    )
    return source, out, args


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@pytest.mark.parametrize("channel", ["ch1", "ch2", "ch3", "ch4"])
@pytest.mark.parametrize("time_format", ["bmjd", "bjd"])
def test_individual_correction_preview_apply_and_exclusive_reuse(
    tmp_path, channel, time_format
):
    source, out, args = correction_inputs(tmp_path, channel, time_format)
    before = {str(p): p.read_bytes() for p in source.iterdir()}
    preview = correct_products(**args)
    assert not out.exists() and not preview["applied"]
    out.mkdir()
    report = correct_products(**args, apply=True)
    assert json.loads(json.dumps(report)) == json.loads(args["audit"].read_text())
    for key in ("input_sha256", "output_sha256"):
        for path, expected in report[key].items():
            assert digest(path) == expected
    for kind, columns in [("full", [1]), ("binned", [2, 3])]:
        original = np.loadtxt(args[kind])
        corrected = np.loadtxt(args["output_" + kind])
        np.testing.assert_array_equal(
            np.delete(original, columns, axis=1), np.delete(corrected, columns, axis=1)
        )
        np.testing.assert_array_equal(np.isnan(original), np.isnan(corrected))
        metadata = read_timing(args["output_" + kind])
        assert metadata["product_role"] == "timing_corrected"
        assert metadata["format"] == time_format.upper() and metadata["scale"] == "TDB"
    breaks = load_aor_breaks(args["output_full"], out, "r", ["r1", "r2"])
    np.testing.assert_allclose(breaks, report["aor_breaks_bmjd"], rtol=0, atol=2.4e-10)
    saved = {str(p): p.read_bytes() for p in out.iterdir()}
    with pytest.raises(FileExistsError):
        correct_products(**args, apply=True)
    assert saved == {str(p): p.read_bytes() for p in out.iterdir()}
    assert before == {str(p): p.read_bytes() for p in source.iterdir()}
    with pytest.raises(ValueError, match="protected"):
        assert_reduction_tree(out, recursive=True)


@pytest.mark.parametrize(
    "indicator",
    ["marker", "audit_directory", "old_manifest", "nested_marker", "relocated_table"],
)
def test_corrected_tree_detection_is_independent_of_manifest_schema(
    tmp_path, indicator
):
    root = tmp_path / "tree"
    root.mkdir()
    if indicator == "marker":
        (root / CORRECTED_TREE_MARKER).write_text("{}")
    elif indicator == "audit_directory":
        (root / "timing_audits").mkdir()
    elif indicator == "old_manifest":
        (root / "observation_corrected_BMJD_TDB.json").write_text("{}")
    elif indicator == "nested_marker":
        (root / "archive").mkdir()
        (root / "archive" / CORRECTED_TREE_MARKER).write_text("{}")
    else:
        _, out, args = correction_inputs(tmp_path)
        out.mkdir()
        correct_products(**args, apply=True)
        (root / "relocated.dat").write_bytes(args["output_full"].read_bytes())
    before = {str(p): p.read_bytes() for p in root.rglob("*") if p.is_file()}
    with pytest.raises(ValueError, match="protected"):
        assert_reduction_tree(root, recursive=True)
    assert before == {str(p): p.read_bytes() for p in root.rglob("*") if p.is_file()}


def test_unmarked_native_tree_is_available_for_reduction(tmp_path):
    (tmp_path / "native.dat").write_text("# legacy/unversioned\n1 55000\n")
    assert_reduction_tree(tmp_path, recursive=True)


@pytest.mark.parametrize("relocated", [False, True])
@pytest.mark.parametrize("recursive", [False, True])
def test_direct_corrected_file_is_protected_without_tree_marker(
    tmp_path, relocated, recursive
):
    _, out, args = correction_inputs(tmp_path)
    out.mkdir()
    correct_products(**args, apply=True)
    path = args["output_full"]
    if relocated:
        destination = tmp_path / "unmarked"
        destination.mkdir()
        moved = destination / "renamed.dat"
        path.rename(moved)
        path = moved
    assert not (path.parent / CORRECTED_TREE_MARKER).exists()
    assert not (path.parent / "timing_audits").exists()
    before = path.read_bytes()
    with pytest.raises(ValueError, match="Corrected product is protected"):
        assert_reduction_tree(path, recursive=recursive)
    assert path.read_bytes() == before


@pytest.mark.parametrize("versioned", [False, True])
@pytest.mark.parametrize("recursive", [False, True])
def test_direct_uncorrected_file_is_available_for_reduction(
    tmp_path, versioned, recursive
):
    path = tmp_path / "native.dat"
    header = timing_header() if versioned else "legacy/unversioned"
    np.savetxt(path, [[1, 55000]], header=header)
    before = path.read_bytes()
    assert_reduction_tree(path, recursive=recursive)
    assert path.read_bytes() == before
