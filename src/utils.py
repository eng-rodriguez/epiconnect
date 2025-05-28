import mne

from mne.io import BaseRaw


# Reduce verbosity of MNE output
mne.set_log_level("WARNING")


def save_raw(raw: BaseRaw, filename: str, overwrite: bool = True) -> None:
    """Save an MNE Raw object to a .fif file."""
    if not isinstance(raw, BaseRaw):
        raise TypeError("Expected 'raw' to be an instance of mne.io.BaseRaw")
    raw.save(filename, overwrite=overwrite)


def load_raw(filename: str, preload: bool = True, verbose: bool = False) -> BaseRaw:
    """Load an MNE Raw object from a .fif file."""
    return mne.io.read_raw_fif(filename, preload=preload, verbose=verbose)
