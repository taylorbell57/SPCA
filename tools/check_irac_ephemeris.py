"""Offline comparison of BCD states against cached geometric Horizons vectors.

The stored four-vector fixture was queried for Spitzer (-79), SSB center (@0),
TDB epochs, Earth/ICRF reference plane, and geometric (no aberration) positions.
Header records use arbitrary IDs linked explicitly by the vectors' header_id
column; source_bcd preserves archive provenance. Coordinate offsets are supplied
by the caller, independent of any target name or observing campaign.
This check does not download ephemerides or alter timing products.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord, get_body_barycentric
from astropy.table import Table
from astropy.time import Time


def compare(headers, vectors, target_coord):
    entries = json.loads(Path(headers).read_text())
    table = Table.read(vectors)
    target = SkyCoord(*target_coord, unit="deg").cartesian.xyz.value
    records = []
    for row in table:
        matches = [
            e
            for e in entries
            if e["id"] == row["header_id"]
            and abs(2451545.0 + e["header"]["ET_OBS"] / 86400 - row["datetime_jd"])
            < 1e-8
        ]
        if len(matches) != 1:
            raise ValueError("Each cached vector must match exactly one header epoch")
        entry = matches[0]
        h = entry["header"]
        t = Time(2451545.0, h["ET_OBS"] / 86400, format="jd", scale="tdb")
        position = np.array([h["SPTZR_" + a] for a in "XYZ"])
        position += get_body_barycentric("sun", t, ephemeris="builtin").xyz.to_value(
            u.km
        )
        horizons = np.array([row[a] for a in "xyz"]) * u.au.to(u.km)
        difference = position - horizons
        requested = SkyCoord(
            h["RA_RQST"], h["DEC_RQST"], unit="deg"
        ).cartesian.xyz.value
        records.append(
            dict(
                id=entry["id"],
                channel=entry["channel"],
                epoch_tdb_jd=float(row["datetime_jd"]),
                position_difference_km=difference.tolist(),
                target_delay_difference_seconds=float(difference @ target / 299792.458),
                differential_direction_difference_seconds=float(
                    difference @ (target - requested) / 299792.458
                ),
            )
        )
    return dict(
        input_sha256={
            str(p): hashlib.sha256(Path(p).read_bytes()).hexdigest()
            for p in (headers, vectors)
        },
        comparisons=records,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--headers", type=Path, required=True)
    parser.add_argument("--vectors", type=Path, required=True)
    parser.add_argument("--target-coord", type=float, nargs=2, required=True)
    print(json.dumps(compare(**vars(parser.parse_args())), indent=2))
