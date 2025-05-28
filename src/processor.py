from typing import List, Optional
from dataclasses import dataclass, field

import mne

# Reduce verbosity of MNE output
mne.set_log_level("WARNING")


@dataclass
class EEGProcessingConfig:
    """Configuration for EEG preprocessing."""

    high_pass: Optional[float] = 1.0
    low_pass: Optional[float] = 80.0
    notch_freqs: List[float] = field(default_factory=lambda: [60.0, 120.0])
    picks: str = "eeg"
    fir_design: str = "firwin"
    reference: str = "average"
    projection: bool = False


class EEGProcessor:
    """Processor for standard EEG preprocessing using MNE."""

    def __init__(self, config: EEGProcessingConfig):
        """Initialize EEGProcessor with the given configuration."""
        self._config = config

    def _apply_bandpass_filter(self, raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
        """Apply high-pass and low-pass filtering to the EEG data."""
        raw.filter(
            l_freq=self._config.high_pass,
            h_freq=self._config.low_pass,
            picks=self._config.picks,
            fir_design=self._config.fir_design,
            skip_by_annotation="edge",
        )
        return raw

    def _apply_notch_filter(self, raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
        """Apply notch filtering to remove powerline and harmonic noise."""
        raw.notch_filter(
            freqs=self._config.notch_freqs,
            picks=self._config.picks,
            filter_length="auto",
            method="spectrum_fit",
        )
        return raw

    def _apply_reference(self, raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
        """Apply EEG referencing."""
        raw.set_eeg_reference(
            self._config.reference, projection=self._config.projection
        )
        return raw

    def process(self, raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
        """Execute the full EEG preprocessing pipeline."""
        processed = raw.copy().load_data()
        processed = self._apply_bandpass_filter(processed)
        processed = self._apply_notch_filter(processed)
        processed = self._apply_reference(processed)
        return processed
