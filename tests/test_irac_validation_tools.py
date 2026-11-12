"""Offline reusable checks with committed ephemerides and generated BCD images."""

from pathlib import Path

import pytest
from conftest import SYNTHETIC_TARGET, make_bcd

from tools.check_irac_ephemeris import compare
from tools.validate_irac_samples import validate

FIXTURES = Path(__file__).parent / "fixtures"


def test_cached_horizons_checker_matches_explicit_header_ids():
    """Associate cached Horizons vectors by header identity and reproduce direction
    comparisons.
    """
    result = compare(
        FIXTURES / "irac_real_headers.json",
        FIXTURES / "horizons_spitzer.ecsv",
        (120.0, -20.0),
    )
    assert len(result["comparisons"]) == 4
    assert all("position_difference_km" in row for row in result["comparisons"])
    assert len(result["input_sha256"]) == 2


def test_local_bcd_checker_accepts_arbitrary_channels_and_coordinates(tmp_path):
    """Validate local BCD samples across channels with explicitly supplied target
    coordinates.
    """
    make_bcd(tmp_path, "ch1")
    make_bcd(tmp_path, "ch4", subarray=True)
    output = tmp_path / "validation.json"
    records = validate(
        tmp_path, output, (SYNTHETIC_TARGET.ra.deg, SYNTHETIC_TARGET.dec.deg)
    )
    assert {r["channel"] for r in records} == {"ch1", "ch4"}
    assert sorted(r["frames"] for r in records) == [1, 64]
    assert output.exists()
    with pytest.raises(FileExistsError):
        validate(tmp_path, output, (120.0, -20.0))
