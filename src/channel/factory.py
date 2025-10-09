from typing import Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.io import loadmat

from channel.base import AWGNModel, ChannelModel, NoNoiseModel
from channel.enums import ChannelType, NoiseType

NOISE_MAP = {
    NoiseType.AWGN: AWGNModel,
    NoiseType.NONE: NoNoiseModel,
}


class ChannelFactory:
    """Factory class to create channel models."""

    @staticmethod
    def create_channel(
        channel_type: ChannelType,
        noise_type: NoiseType,
        snr_db: float,
        time_response: Optional[NDArray[np.complex128]] = None,
    ) -> ChannelModel:
        """Create a channel model based on the specified type and parameters."""
        if channel_type == ChannelType.FLAT:
            noise_model = NOISE_MAP[noise_type]()
            return ChannelModel(
                impulse_response=np.array(
                    [1.0], dtype=np.complex128
                ),  # Flat channel impulse response
                noise_model=noise_model,
                snr_db=snr_db,
                channel_type=ChannelType.FLAT,
            )

        if channel_type == ChannelType.CUSTOM:
            if time_response is None:
                raise ValueError("Time response must be provided for custom channels.")
            noise_model = NOISE_MAP[noise_type]()
            return ChannelModel(
                impulse_response=time_response,
                noise_model=noise_model,
                snr_db=snr_db,
                channel_type=ChannelType.CUSTOM,
            )

        raise ValueError(f"Unsupported channel type: {channel_type}")

    @classmethod
    def create_flat_channel(cls, snr_db: float, noise_type: NoiseType) -> Tuple[ChannelModel, int]:
        """Create a flat channel with the specified SNR and noise type."""
        return (
            cls.create_channel(
                channel_type=ChannelType.FLAT,
                noise_type=noise_type,
                snr_db=snr_db,
            ),
            1,
        )  # Flat channel has impulse response length 1

    @classmethod
    def create_custom_channel(
        cls,
        snr_db: float,
        noise_type: NoiseType,
        time_response: NDArray[np.complex128],
    ) -> ChannelModel:
        """Create a custom channel with the specified SNR, noise type, and time response."""
        return cls.create_channel(
            channel_type=ChannelType.CUSTOM,
            noise_type=noise_type,
            snr_db=snr_db,
            time_response=time_response,
        )

    EXTENSION_LOADER_MAP = {
        "mat": loadmat,
        "txt": np.loadtxt,
        "npy": np.load,
    }

    DATA_CLEANUP_MAP = {
        "mat": lambda data: np.array(data["h"]).ravel(),
        "txt": lambda data: np.array(data).ravel(),
        "npy": lambda data: data.ravel(),
    }

    @classmethod
    def create_custom_channel_from_file(
        cls,
        snr_db: float,
        noise_type: NoiseType,
        filepath: str,
    ) -> Tuple[ChannelModel, int]:
        """Create a custom channel by loading the magnitude response from a file."""
        extension = filepath.split(".")[-1].lower()

        time_response_loader = cls.EXTENSION_LOADER_MAP.get(extension)
        if time_response_loader is None:
            raise ValueError(f"Unsupported file extension: {extension}")

        data = time_response_loader(filepath)
        time_response = cls.DATA_CLEANUP_MAP[extension](data)

        return (
            cls.create_custom_channel(
                snr_db=snr_db,
                noise_type=noise_type,
                time_response=time_response,
            ),
            time_response.size - 1,  # order
        )
