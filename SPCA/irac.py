"""IRAC channel identities and SPCA's historical aperture-directory names."""

CHANNELS = ("ch1", "ch2", "ch3", "ch4")
WAVELENGTH_UM = dict(zip(CHANNELS, (3.6, 4.5, 5.8, 8.0)))
# ch1/ch2 are existing on-disk names. ch3/ch4 extend that integer-wavelength
# convention; earlier SPCA incorrectly sent every non-ch1 channel to 4um.
APERTURE_PREFIX = dict(zip(CHANNELS, ("3um", "4um", "5um", "8um")))
DETECTOR = dict(zip(CHANNELS, ("InSb", "InSb", "Si:As", "Si:As")))


def validate_channel(channel):
    """Accept explicit channel names, never silently fall back to ch1/ch2."""
    if channel not in CHANNELS:
        raise ValueError(f"Invalid IRAC channel {channel!r}; choose {CHANNELS}")
    return channel


def aperture_prefix(channel):
    return APERTURE_PREFIX[validate_channel(channel)]


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
