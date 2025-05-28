from pathlib import Path
from typing import Optional, Sequence, Union

import mne
import numpy as np
import pandas as pd


class Dataloader:
    """
    Class for loading EEG data and creating MNE Raw objects.
    """

    DEFAULT_CHANNEL_NAMES: Sequence[str] = [
        "C3",
        "C4",
        "O1",
        "O2",
        "Cz",
        "F3",
        "F4",
        "F7",
        "F8",
        "Fp1",
        "Fp2",
        "P3",
        "P4",
        "Pz",
        "T3",
        "T4",
        "T5",
        "T6",
    ]

    def __init__(
        self,
        sampling_rate: float = 256,
        channel_names: Optional[Sequence[str]] = None,
        log_level: str = "WARNING",
    ) -> None:
        """
        Initialize the EEG loader.
        """
        mne.set_log_level(log_level)
        self.sampling_rate: float = sampling_rate
        self.channel_names: list[str] = (
            list(channel_names)
            if channel_names is not None
            else list(self.DEFAULT_CHANNEL_NAMES)
        )
        self.raw_data: Optional[np.ndarray] = None
        self.actual_channel_names: Optional[list[str]] = None
        self._mne_raw: Optional[mne.io.BaseRaw] = None

    def load_data(
        self,
        filepath: Union[str, Path],
        delimeter: str = "\t",
        header: Optional[Union[int, Sequence[int]]] = None,
        drop_columns: Optional[Sequence[int]] = None,
        channel_names: Optional[Sequence[str]] = None,
    ) -> np.ndarray:
        """
        Load EEG data from a plain text file.
        """
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        if header is not None:
            df = pd.read_csv(path, delimiter=delimeter, header=header)
        else:
            df = pd.read_csv(path, delimiter=delimeter, header=None)

        if drop_columns:
            df = df.drop(df.columns[list(drop_columns)], axis=1)

        # Transpose so that shape is (channels, samples)
        data_arr = df.to_numpy().T
        self.raw_data = data_arr

        # Set channel names
        names = list(channel_names) if channel_names is not None else self.channel_names

        if len(names) < data_arr.shape[0]:
            raise ValueError(
                f"Not enough channel names provided: {len(names)} < {data_arr.shape[0]}"
            )

        self.actual_channel_names = names[: data_arr.shape[0]]

        return self.raw_data

    def create_mne_raw(
        self,
        ch_types: Union[str, Sequence[str]] = "eeg",
        montage: Union[str, mne.channels.DigMontage] = "standard_1020",
    ) -> mne.io.RawArray:
        """
        Create MNE Raw obejct from loaded EEG data.
        """
        if self.raw_data is None:
            raise ValueError("No data loaded. Call load_data() first.")

        n_channels, _ = self.raw_data.shape

        # Normalize channel types
        if isinstance(ch_types, str):
            ch_types_list = [ch_types] * n_channels
        elif isinstance(ch_types, Sequence):
            if len(ch_types) != n_channels:
                raise ValueError(
                    f"Length of ch_types ({len(ch_types)}) does not match number of channels ({n_channels})."
                )
            ch_types_list = list(ch_types)
        else:
            raise TypeError("ch_types must be a string or sequence of strings.")

        # Prepare info and Raw object
        info = mne.create_info(
            ch_names=self.actual_channel_names,
            sfreq=self.sampling_rate,
            ch_types=ch_types_list,
        )
        raw = mne.io.RawArray(self.raw_data, info)

        # Apply montage
        if isinstance(montage, str):
            montage_obj = mne.channels.make_standard_montage(montage)
        else:
            montage_obj = montage
        raw.set_montage(montage_obj, match_case=False)

        self._mne_raw = raw
        return raw

    @property
    def mne_raw(self) -> Optional[mne.io.BaseRaw]:
        """
        Return the MNE Raw object if created, otherwise None.
        """
        return self._mne_raw
