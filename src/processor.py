from scipy.stats import kurtosis
from typing import List, Optional
from mne.preprocessing import ICA
from dataclasses import dataclass, field

import mne
import numpy as np

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

    def remove_artifacts_ica(
        self,
        raw: mne.io.BaseRaw,
        n_components: Optional[int] = None,
        method: str = "fastica",
        random_state: int = 97,
        manual: bool = False,
        heuristic: bool = True,
    ) -> mne.io.BaseRaw:
        """Remove artifacts using ICA without EOG/ECG channels."""

        raw_copy = raw.copy().load_data()
        picks = mne.pick_types(raw_copy.info, eeg=True, exclude="bads")

        ica = ICA(n_components=n_components, method=method, random_state=random_state)
        ica.fit(raw_copy, picks=picks)

        components_to_exclude = []

        if heuristic:
            # herusitc: detect outliers using kurtosis or low variance
            sources = ica.get_sources(raw_copy).get_data()
            kurtosis_vals = kurtosis(sources, axis=1, fisher=True, bias=False)
            threshold = 5.0
            components_to_exclude = list(np.where(np.abs(kurtosis_vals) > threshold)[0])

        if manual:
            print("Review ICA components. Close all plots to proceed.")
            ica.plot_components()
            ica.plot_sources(raw_copy)

            comp_str = input(
                "Enter the component indices to exclude (comma-separated, e.g., 0,1,4): "
            )
            manual_comps = [
                int(idx.strip()) for idx in comp_str.split(",") if idx.strip().isdigit()
            ]
            components_to_exclude.extend(manual_comps)

        # Mark and apply exclusion
        ica.exclude = components_to_exclude
        raw_cleaned = ica.apply(raw_copy)

        return raw_cleaned
