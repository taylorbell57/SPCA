"""Check supplied local IRAC BCD samples for an arbitrary fixed ICRS target.

No network access or science-product writes. Supply a directory containing
*_bcd.fits, target coordinates in degrees, and a new JSON report path.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS

from SPCA.irac import validate_header_channel
from SPCA.irac_timing import exposure_midpoints, target_correction
from SPCA.Photometry_Common import full_frame_crop, prepare_image


def validate(samples, output, target_coord):
    target = SkyCoord(*target_coord, unit="deg", frame="icrs")
    paths = sorted(Path(samples).rglob("*_bcd.fits"))
    if not paths:
        raise ValueError("No local BCD samples found")
    if Path(output).exists():
        raise FileExistsError("Choose a new report path")
    records = []
    for source in paths:
        original = source.read_bytes()
        header = fits.getheader(source)
        channel = "ch" + str(int(header["CHNLNUM"]))
        validate_header_channel(header, channel)
        prepared = prepare_image(
            str(source.parent),
            "r",
            [str(source)],
            [1],
            target_coord=target,
            channel=channel,
        )
        image = fits.getdata(source)
        if image.ndim == 2:
            expected = image[full_frame_crop(header, target)][None]
        else:
            expected = image
        factor = header["GAIN"] * header["EXPTIME"] / header["FLUXCONV"]
        np.testing.assert_allclose(
            prepared[:, :-1].reshape(expected.shape), expected * factor
        )
        times = exposure_midpoints(header, target_coord=target)
        np.testing.assert_array_equal(prepared[:, -1], times)
        assert original == source.read_bytes()
        records.append(
            dict(
                file=str(source),
                channel=channel,
                sha256=hashlib.sha256(original).hexdigest(),
                frames=len(times),
                target_pixel=[
                    float(v) for v in WCS(header, naxis=2).world_to_pixel(target)
                ],
                first_midpoint_bmjd_tdb=float(times[0]),
                direction_correction_seconds=float(
                    target_correction(header, target, [0])[0]
                ),
            )
        )
    with Path(output).open("x") as stream:
        json.dump(records, stream, indent=2)
        stream.write("\n")
    return records


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target-coord", type=float, nargs=2, required=True)
    print(f"Verified {len(validate(**vars(parser.parse_args())))} BCD samples")
