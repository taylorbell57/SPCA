"""Validate and copy legacy SPCA aperture products with corrected timestamps.

No in-place mode. Header inventories contain values, not paths to be followed.
Use ``python -m SPCA.timing_correction --help``. All writes require --apply;
without it, correction is a report-only operation, including bin validation.
"""

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits

from . import irac_timing
from .irac import CHANNELS, validate_header_channel
from .irac_timing import exposure_midpoints, frame_count, read_timing, timing_metadata

INVENTORY_FORMAT = "spca-irac-headers-v1"
TIME_ATOL_DAYS = 2e-10  # 17 microseconds; allows text/float rounding, not bad cadence
HEADER_KEYS = (
    "NAXIS",
    "NAXIS1",
    "NAXIS2",
    "NAXIS3",
    "CHNLNUM",
    "AORKEY",
    "EXPID",
    "DCENUM",
    "BMJD_OBS",
    "MJD_OBS",
    "UTCS_OBS",
    "ET_OBS",
    "FRAMTIME",
    "EXPTIME",
    "DATE_OBS",
    "REQTYPE",
    "SAMPTIME",
    "AFOWLNUM",
    "ATIMEEND",
    "AINTBEG",
    "CREATOR",
    "AORTIME",
    "RA_RQST",
    "DEC_RQST",
    "CRVAL1",
    "CRVAL2",
    "SPTZR_X",
    "SPTZR_Y",
    "SPTZR_Z",
    "SPTZR_VX",
    "SPTZR_VY",
    "SPTZR_VZ",
    "SPTZR_LT",
)


CORRECTED_TREE_MARKER = ".spca-corrected-products.json"


def assert_reduction_tree(path, recursive=False):
    """Protect corrected trees, including earlier exports without root markers.

    Standard reduction/resumption must not regenerate legacy-style manifests
    here. Cleanup also checks nested exports and relocated corrected tables.
    Missing or corrupt marker content does not remove protection. Callers must
    invoke this guard themselves before destructive work; it does not intercept
    filesystem operations. A directly supplied file is checked using its timing
    metadata, even outside a marked tree. Use recursive=True to inspect tables
    inside directories without root markers or audit subdirectories.
    """
    path = Path(path)
    for parent in (path, *path.parents):
        if (
            (parent / CORRECTED_TREE_MARKER).exists()
            or (parent / "timing_audits").is_dir()
            or next(parent.glob("*_corrected_*_TDB.json"), None) is not None
        ):
            raise ValueError(
                f"Corrected products are protected from reduction/cleanup: {parent}"
            )
    if path.is_file():
        try:
            corrected = read_timing(path).get("product_role") == "timing_corrected"
        except (ValueError, OSError):
            corrected = False
        if corrected:
            raise ValueError(f"Corrected product is protected from cleanup: {path}")
    if recursive and path.is_dir():
        for pattern in (
            CORRECTED_TREE_MARKER,
            "*_corrected_*_TDB.json",
            "timing_audits",
        ):
            if next(path.rglob(pattern), None) is not None:
                raise ValueError(
                    f"Corrected products are protected from cleanup: {path}"
                )
        for table in path.rglob("*.dat"):
            if table.is_symlink():
                continue
            try:
                corrected = read_timing(table).get("product_role") == "timing_corrected"
            except (ValueError, OSError):
                continue
            if corrected:
                raise ValueError(
                    f"Corrected product is protected from cleanup: {table}"
                )


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def make_inventory(data_dir, channel, aors, expected_bcd):
    """Read only explicitly selected local BCD headers; never follow manifests."""
    root = Path(data_dir)
    if root.name != channel or not aors or len(set(aors)) != len(aors):
        raise ValueError("Use data/chN and distinct explicit science AOR names")
    entries, counts = [], {}
    for aor in sorted(aors):
        if not re.fullmatch(r"r[0-9]+", aor):
            raise ValueError(f"Invalid AOR: {aor}")
        files = sorted((root / aor / channel / "bcd").glob("*_bcd.fits"))
        if not files:
            raise ValueError(f"No BCDs for {aor}")
        counts[aor] = len(files)
        for path in files:
            header = fits.getheader(path)
            validate_header_channel(header, channel)
            if "AORKEY" in header and int(header["AORKEY"]) != int(aor[1:]):
                raise ValueError(f"AORKEY mismatch: {path}")
            exposure_midpoints(header)  # fail before writing an unusable inventory
            entries.append(
                {
                    "id": f"{aor}/{path.name}",
                    "aor": aor,
                    "header": {
                        key: header[key] for key in HEADER_KEYS if key in header
                    },
                }
            )
    if len(entries) != expected_bcd:
        raise ValueError(f"Expected {expected_bcd} BCDs; found {len(entries)}")
    return {
        "format": INVENTORY_FORMAT,
        "channel": channel,
        "aor_counts": counts,
        "bcd_count": len(entries),
        "entries": entries,
    }


def reconstruct(inventory, time_scale="tdb", target_coord=None, time_format="bmjd"):
    """Reconstruct legacy row identities and corrected times from each header."""
    if inventory.get("format") != INVENTORY_FORMAT:
        raise ValueError(f"Expected {INVENTORY_FORMAT} inventory")
    entries = inventory["entries"]
    if not entries or len(entries) != inventory["bcd_count"]:
        raise ValueError("Empty or incomplete header inventory")
    if len({entry["id"] for entry in entries}) != len(entries):
        raise ValueError("Duplicate BCD identities in header inventory")
    old, new, counts = [], [], {}
    for entry in entries:
        header = entry["header"]
        validate_header_channel(header, inventory["channel"])
        counts[entry["aor"]] = counts.get(entry["aor"], 0) + 1
        if "AORKEY" in header and int(header["AORKEY"]) != int(entry["aor"][1:]):
            raise ValueError("Header AORKEY does not match inventory")
        n = frame_count(header)
        step = float(header["FRAMTIME"]) / 86400
        start = float(header["BMJD_OBS"])
        old.extend(np.linspace(start + step / 2, start + (n - 1) * step, n))
        new.extend(exposure_midpoints(header, time_scale, target_coord, time_format))
    if counts != inventory["aor_counts"]:
        raise ValueError("AOR counts do not match inventory")
    old, new = np.asarray(old), np.asarray(new)
    order = np.argsort(old, kind="stable")
    if not np.array_equal(order, np.argsort(new, kind="stable")):
        raise ValueError("Correction changes row order; regenerate photometry")
    old, new = old[order], new[order]
    if np.any(np.diff(old) <= 0) or np.any(np.diff(new) <= 0):
        raise ValueError("Duplicate/overlapping exposure times; cannot identify rows")
    return old, new


def _require_close(actual, expected, label, atol=TIME_ATOL_DAYS):
    if actual.shape != expected.shape or not np.allclose(
        actual, expected, rtol=0, atol=atol, equal_nan=True
    ):
        raise ValueError(f"{label} does not match reconstructed legacy products")


def _bin_times(times, good, size):
    rows = []
    for start in range(0, len(times), size):
        values = times[start : start + size][good[start : start + size]]
        rows.append(
            (np.median(values), np.std(values)) if len(values) else (np.nan, np.nan)
        )
    return np.asarray(rows)


def _table(path, columns):
    payload = Path(path).read_bytes()
    if b"SPCA_TIMING=" in payload or b"SPCA_TIMING_JSON=" in payload:
        raise ValueError(f"Already versioned timestamps; refusing to recorrect {path}")
    data = np.loadtxt(payload.decode("utf-8").splitlines(), ndmin=2)
    if data.shape[1] != columns:
        raise ValueError(f"Expected {columns} columns in {path}")
    return payload, data


def _replace_columns(payload, replacements, metadata):
    """Preserve all non-time tokens, whitespace, row order, and NaN masks."""
    result, row = [], 0
    for line in payload.decode("utf-8").splitlines(keepends=True):
        if not line.strip() or line.lstrip().startswith("#"):
            result.append(line)
            continue
        tokens = list(re.finditer(r"\S+", line.split("#", 1)[0]))
        for column in sorted(replacements, reverse=True):
            match = tokens[column]
            value = replacements[column][row]
            line = line[: match.start()] + f"{value:.18e}" + line[match.end() :]
        result.append(line)
        row += 1
    prefix = (
        f"# SPCA_TIMING={metadata['version']}; "
        f"{metadata['format']}_{metadata['scale']}; midpoint\n"
        "# SPCA_TIMING_JSON=" + json.dumps(metadata, sort_keys=True) + "\n"
    )
    return (prefix + "".join(result)).encode("utf-8")


def correct_products(
    headers,
    full,
    binned=None,
    bin_size=None,
    time_scale="tdb",
    output_full=None,
    output_binned=None,
    audit=None,
    apply=False,
    output_breaks=None,
    target_coord=None,
    time_format="bmjd",
    source_convention="legacy-spca-bmjd-utc",
):
    """Validate legacy aperture rows/bins before writing separate audited copies.

    Only unversioned legacy SPCA linspace BMJD_UTC tables are supported: full
    tables have 8 columns (time at index 1), bins 16 (time/scatter at 2/3).
    Complete spca-irac-headers-v1 inventories must identify every source BCD
    and its channel/AOR. Source timestamps must match at 2e-10 days, with zero
    relative tolerance. Bin masks are reconstructed from timestamps/scatter
    and validated against the corresponding fluxes; ambiguities are errors.

    Default output is BMJD_TDB. A target adds a header-state differential
    direction correction; omission explicitly retains the header reference.
    BJD output changes only the format. Non-time tokens and NaN masks are
    preserved byte-for-byte. --apply requires distinct, nonexistent outputs
    and an audit path; an existing output is never implicitly resumed/replaced.
    Audits record conventions, header/code/product hashes and correction sizes.
    """
    if source_convention != "legacy-spca-bmjd-utc":
        raise ValueError("Only explicitly identified legacy SPCA BMJD_UTC is supported")
    if target_coord is not None and not isinstance(target_coord, SkyCoord):
        target_coord = SkyCoord(*target_coord, unit="deg", frame="icrs")
    metadata = timing_metadata(time_scale, target_coord, time_format)
    metadata.update(
        bin_size=bin_size,
        bin_mask="pending_validation",
        product_role="timing_corrected",
        source_convention=source_convention,
    )
    header_bytes = Path(headers).read_bytes()
    inventory = json.loads(header_bytes)
    old, new = reconstruct(inventory, time_scale, target_coord, time_format)
    format_offset = irac_timing.JD_MJD_OFFSET if time_format == "bjd" else 0
    full_bytes, data = _table(full, 8)
    if len(data) != len(old) or not np.isfinite(data[:, 1]).all():
        raise ValueError("Full row count/finite timestamps do not match inventory")
    _require_close(data[:, 1], old, "Full timestamps")
    inputs = {Path(headers): header_bytes, Path(full): full_bytes}
    outputs = [(output_full, _replace_columns(full_bytes, {1: new}, metadata))]
    mask_policy, bin_delta = None, None
    if binned is not None:
        if bin_size is None or bin_size < 1 or int(bin_size) != bin_size:
            raise ValueError("Binned correction requires explicit positive bin_size")
        bin_bytes, bins = _table(binned, 16)
        inputs[Path(binned)] = bin_bytes
        if len(bins) != (len(data) + bin_size - 1) // bin_size:
            raise ValueError("Binned row count does not match bin_size")
        good_flux = np.isfinite(data[:, 0])
        # Older SPCA versions binned times before masking flux. Identify the
        # policy from BOTH time and scatter, never assume a constant shift.
        policies = {"finite_flux": good_flux, "all_rows": np.ones(len(data), bool)}
        candidates = []
        for name, good in policies.items():
            expected = _bin_times(old, good, bin_size)
            if np.allclose(
                bins[:, 2:4], expected, rtol=0, atol=TIME_ATOL_DAYS, equal_nan=True
            ):
                candidates.append((name, _bin_times(new, good, bin_size)))
        if not candidates:
            raise ValueError("Cannot reconstruct binned time masks/statistics")
        mask_policy, corrected_bins = candidates[0]
        for _, alternate in candidates[1:]:
            _require_close(alternate, corrected_bins, "Ambiguous bin masks")
        # Validate that these are corresponding full/binned flux products.
        flux_medians = _bin_times(data[:, 0], good_flux, bin_size)[:, 0]
        finite = np.isfinite(bins[:, 0])
        if not np.allclose(bins[finite, 0], flux_medians[finite], rtol=1e-10, atol=0):
            raise ValueError("Binned flux does not match full-resolution product")
        metadata["bin_mask"] = mask_policy
        metadata["binning"] = (
            "median of surviving flux timestamps"
            if mask_policy == "finite_flux"
            else "median of all legacy time rows, independent of flux masks"
        )
        bin_delta = (corrected_bins[:, 0] - format_offset - bins[:, 2]) * 86400
        outputs.append(
            (
                output_binned,
                _replace_columns(
                    bin_bytes,
                    {2: corrected_bins[:, 0], 3: corrected_bins[:, 1]},
                    metadata,
                ),
            )
        )
    elif output_binned is not None:
        raise ValueError("output_binned requires a binned input")

    outputs[0] = (output_full, _replace_columns(full_bytes, {1: new}, metadata))
    starts = {}
    for entry in inventory["entries"]:
        first = float(
            exposure_midpoints(entry["header"], time_scale, target_coord, time_format)[
                0
            ]
        )
        starts[entry["aor"]] = min(starts.get(entry["aor"], first), first)
    breaks = sorted(starts.values())[1:]
    if output_breaks is not None:
        outputs.append(
            (
                output_breaks,
                (
                    "# SPCA_TIMING_JSON="
                    + json.dumps({**metadata, "aor_starts": starts}, sort_keys=True)
                    + "\n"
                    + " ".join(f"{value:.15f}" for value in breaks)
                    + "\n"
                ).encode(),
            )
        )

    delta = (new - format_offset - old) * 86400

    def stats(values):
        valid = values[np.isfinite(values)]
        return (
            {
                "min": float(valid.min()),
                "median": float(np.median(valid)),
                "max": float(valid.max()),
            }
            if len(valid)
            else None
        )

    _, midpoint_utc = reconstruct(inventory, "utc")
    header_tdb = None
    if time_scale == "tdb":
        _, header_tdb = reconstruct(inventory, "tdb")
    components = {
        "midpoint_cadence_repair": stats((midpoint_utc - old) * 86400),
        "utc_to_tdb": (
            stats((header_tdb - midpoint_utc) * 86400)
            if header_tdb is not None
            else None
        ),
        "target_direction_and_lighttime_rate": (
            stats(
                (
                    new
                    - format_offset
                    - (header_tdb if time_scale == "tdb" else midpoint_utc)
                )
                * 86400
            )
            if target_coord is not None
            else None
        ),
    }
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "applied": False,
        "source_convention": "legacy SPCA linspace, BMJD_UTC",
        "software_sha256": {
            "timing_correction.py": sha256(Path(__file__).read_bytes()),
            "irac_timing.py": sha256(Path(irac_timing.__file__).read_bytes()),
        },
        "timing": metadata,
        "header_reference_directions_deg": sorted(
            {
                (e["header"].get("RA_RQST"), e["header"].get("DEC_RQST"))
                for e in inventory["entries"]
            },
            key=str,
        ),
        "channel": inventory["channel"],
        "aor_counts": inventory["aor_counts"],
        "rows": len(data),
        "rejected_flux_rows": int(np.sum(~np.isfinite(data[:, 0]))),
        "input_sha256": {
            str(path.absolute()): sha256(blob) for path, blob in inputs.items()
        },
        "full_shift_seconds": stats(delta),
        "component_shift_seconds": components,
        "binned_shift_seconds": stats(bin_delta) if bin_delta is not None else None,
        "binned_mask_policy": mask_policy,
        "aor_starts": starts,
        "aor_start_bmjd": {k: v - format_offset for k, v in starts.items()},
        "aor_breaks_bmjd": [v - format_offset for v in breaks],
        "modified_columns": {"full": [1], "binned": [2, 3] if binned else []},
        "non_time_columns": "unchanged byte-for-byte, including NaN tokens",
    }
    # Dry-run validates supplied destinations too, but creates nothing.
    destinations = [Path(path) for path, _ in outputs if path is not None]
    if audit is not None:
        destinations.append(Path(audit))
    resolved = [path.resolve() for path in destinations]
    if len(set(resolved)) != len(resolved) or set(resolved) & {
        path.resolve() for path in inputs
    }:
        raise ValueError("Inputs, outputs, and audit must be distinct paths")
    if any(path.exists() or path.is_symlink() for path in destinations):
        raise FileExistsError("Refusing to overwrite an output or audit")
    if not apply:
        return report
    if any(path is None for path, _ in outputs) or audit is None:
        raise ValueError("--apply requires explicit output paths and an audit path")
    if any(not path.parent.is_dir() for path in destinations):
        raise ValueError("Create the destination directories before applying")
    for path, blob in inputs.items():
        if path.read_bytes() != blob:
            raise ValueError(f"Input changed during validation: {path}")
    report["applied"] = True
    report["output_sha256"] = {
        str(Path(path).absolute()): sha256(blob) for path, blob in outputs
    }
    outputs.append((audit, (json.dumps(report, indent=2) + "\n").encode()))
    created = []
    try:
        for path, blob in outputs:
            with Path(path).open("xb") as stream:
                created.append(Path(path))
                stream.write(blob)
    except BaseException:
        for path in created:
            path.unlink()
        raise
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    collect = commands.add_parser(
        "inventory", help="Export explicitly selected FITS headers"
    )
    collect.add_argument("--data-dir", type=Path, required=True)
    collect.add_argument("--channel", choices=CHANNELS, required=True)
    collect.add_argument("--aor", action="append", required=True)
    collect.add_argument("--expect-bcd", type=int, required=True)
    collect.add_argument("--output", type=Path, required=True)
    fix = commands.add_parser(
        "correct", help="Validate legacy products; report only by default"
    )
    for name in ("headers", "full"):
        fix.add_argument(f"--{name}", type=Path, required=True)
    for name in ("binned", "output-full", "output-binned", "output-breaks", "audit"):
        fix.add_argument(f"--{name}", type=Path)
    fix.add_argument("--bin-size", type=int)
    fix.add_argument("--time-scale", choices=("utc", "tdb"), default="tdb")
    fix.add_argument("--time-format", choices=("bmjd", "bjd"), default="bmjd")
    fix.add_argument(
        "--target-coord", type=float, nargs=2, metavar=("RA_DEG", "DEC_DEG")
    )
    fix.add_argument(
        "--source-convention",
        choices=("legacy-spca-bmjd-utc",),
        default="legacy-spca-bmjd-utc",
    )
    fix.add_argument("--apply", action="store_true")
    args = vars(parser.parse_args(argv))
    command = args.pop("command")
    try:
        if command == "inventory":
            output = args.pop("output")
            args["aors"] = args.pop("aor")
            args["expected_bcd"] = args.pop("expect_bcd")
            result = make_inventory(**args)
            with output.open("x") as stream:
                json.dump(result, stream, indent=2)
                stream.write("\n")
            print(f"Wrote {result['bcd_count']} headers to {output}")
        else:
            print(json.dumps(correct_products(**args), indent=2))
    except (ValueError, OSError, KeyError) as error:
        parser.exit(2, f"Refused: {error}\n")


if __name__ == "__main__":
    main()
