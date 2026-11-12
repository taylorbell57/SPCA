"""IRAC channel identities and photometry-directory compatibility."""

from pathlib import Path

CHANNELS = ("ch1", "ch2", "ch3", "ch4")
# Nominal band wavelengths in micrometres (IRAC Instrument Handbook, Table 2.1).
WAVELENGTH_UM = dict(zip(CHANNELS, (3.6, 4.5, 5.8, 8.0)))
APERTURE_PREFIX = dict(zip(CHANNELS, CHANNELS))
LEGACY_PREFIX = {"ch1": "3um", "ch2": "4um"}
DETECTOR = dict(zip(CHANNELS, ("InSb", "InSb", "Si:As", "Si:As")))


def validate_channel(channel):
    """Accept explicit channel names, never silently fall back to ch1/ch2."""
    if channel not in CHANNELS:
        raise ValueError(f"Invalid IRAC channel {channel!r}; choose {CHANNELS}")
    return channel


def aperture_prefix(channel):
    """Return the channel prefix used for all newly generated photometry."""
    return APERTURE_PREFIX[validate_channel(channel)]


def photometry_prefixes(channel):
    """Readable prefixes, including existing ch1/ch2 wavelength names."""
    validate_channel(channel)
    return (channel, LEGACY_PREFIX[channel]) if channel in LEGACY_PREFIX else (channel,)


def resolve_photometry_dir(parent, channel, suffix, filename):
    """Find a dataset in either convention; refuse ambiguous coexisting copies.

    Explicit paths in bestPhOption.txt bypass this lookup. Neither convention
    takes precedence when both contain the requested dataset.
    """
    candidates = [
        Path(parent) / (prefix + suffix)
        for prefix in photometry_prefixes(channel)
        if (Path(parent) / (prefix + suffix) / filename).is_file()
    ]
    if len(candidates) > 1:
        raise ValueError(f"Ambiguous photometry for {filename}: {candidates}")
    if not candidates:
        raise FileNotFoundError(f"No {channel}{suffix} photometry for {filename}")
    return str(candidates[0]) + "/"


def validate_header_channel(header, channel):
    """Check archive channel identity when it is supplied in a FITS header."""
    validate_channel(channel)
    if "CHNLNUM" in header and int(header["CHNLNUM"]) != int(channel[-1]):
        raise ValueError(f"CHNLNUM={header['CHNLNUM']} does not match {channel}")


def aor_tag(snippet):
    """Photometry filename tag: r41587 or 41587 -> 41587, never r41587.

    The historical all-AOR selector 'r' maps to the empty filename tag.
    """
    if not isinstance(snippet, str) or not snippet:
        raise ValueError("An explicit AOR snippet/tag is required")
    tag = snippet[1:] if snippet.startswith("r") else snippet
    if tag and (not tag.isascii() or not tag.isdigit()):
        raise ValueError(f"Invalid AOR snippet/tag: {snippet!r}")
    return tag


def aor_breaks_name(snippet):
    return f"aorBreaks_AORs{aor_tag(snippet)}.txt"
