"""Offline tests must never read production FITS data."""

import os
import tempfile

import pytest
from astropy.coordinates import SkyCoord
from astropy.io import fits

SYNTHETIC_TARGET = SkyCoord(120.0, -20.0, unit="deg")


@pytest.fixture(autouse=True)
def local_fits_only(monkeypatch):
    """All image inputs are disposable fixtures under a local temp directory.

    Compare path strings before any file opening. File objects (e.g. BytesIO)
    remain usable for archive-unit tests. No production path is resolved.
    """
    roots = (
        "/tmp/",
        "/private/tmp/",
        "/var/folders/",
        "/private/var/folders/",
        os.path.abspath(tempfile.gettempdir()) + os.sep,
    )

    def wrap(function):
        def checked(path, *args, **kwargs):
            if isinstance(path, (str, bytes, os.PathLike)):
                name = os.path.abspath(os.fsdecode(path))
                if not name.startswith(roots):
                    raise AssertionError(f"Tests may only open temporary FITS: {name}")
            return function(path, *args, **kwargs)

        return checked

    monkeypatch.setattr(fits, "open", wrap(fits.open))
    monkeypatch.setattr(fits, "getheader", wrap(fits.getheader))
    monkeypatch.setattr(fits, "getdata", wrap(fits.getdata))


def add_synthetic_state(header, target):
    """A controlled, self-consistent clock/vector fixture, NOT a mission orbit."""
    import numpy as np

    from SPCA.irac_timing import C_KM_S, _sun_state

    header.setdefault("ET_OBS", 300000066.184)
    header.setdefault("UTCS_OBS", header["ET_OBS"] - 66.184)
    sun, velocity = _sun_state(header["ET_OBS"])
    pos = np.array([150000000.0, 0.0, 0.0])
    for i, axis in enumerate("XYZ"):
        header["SPTZR_" + axis] = pos[i]
        header["SPTZR_V" + axis] = -velocity[i]
    header["RA_RQST"], header["DEC_RQST"] = target.ra.deg, target.dec.deg
    header["MJD_OBS"] = (
        header["BMJD_OBS"]
        - np.dot(pos + sun, target.cartesian.xyz.value) / C_KM_S / 86400
    )
    return header


def make_bcd(root, channel, aor="r123", index=0, subarray=False):
    import numpy as np
    from astropy.io import fits
    from astropy.wcs import WCS

    folder = root / f"PID12345_2008/data/{channel}/{aor}/{channel}/bcd"
    folder.mkdir(parents=True, exist_ok=True)
    x0, y0 = (15.3, 15.7) if subarray else (100.3, 153.7)
    w = WCS(naxis=2)
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.wcs.crval = [SYNTHETIC_TARGET.ra.deg, SYNTHETIC_TARGET.dec.deg]
    w.wcs.crpix = [x0 + 1, y0 + 1]
    w.wcs.cdelt = [-1.22 / 3600, 1.22 / 3600]
    header = w.to_header()
    header.update(
        CHNLNUM=int(channel[-1]),
        AORKEY=int(aor[1:]),
        BMJD_OBS=54773.0 + index * (130 if subarray else 12) / 86400,
        FRAMTIME=2 if subarray else 12,
        EXPTIME=1.92 if subarray else 10.4,
        GAIN=3.8,
        FLUXCONV=0.5,
        ET_OBS=278944849.566 + 12 * index,
        UTCS_OBS=278944784.384 + 12 * index,
        DATE_OBS="2008-11-03T00:39:44",
    )
    add_synthetic_state(header, SYNTHETIC_TARGET)
    size = 32 if subarray else 256
    y, x = np.mgrid[:size, :size]
    image = (
        30
        + (-1.0) ** (x + y)
        + (500 + 0.1 * index) * np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * 1.3**2))
    )
    if subarray:
        image = np.repeat(image[None], 64, axis=0)
    path = folder / f"{index:04d}_bcd.fits"
    fits.PrimaryHDU(image, header).writeto(path)
    return path
