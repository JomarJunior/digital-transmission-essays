from abc import ABC, abstractmethod
import numpy as np
from numpy.typing import NDArray
from numpy import complex128
from pydantic import BaseModel
from channel.enums import ChannelType


class INoiseModel(ABC):
    """Interface for noise models."""

    @abstractmethod
    def add_noise(self, signal: NDArray[complex128], snr_db: float) -> NDArray[complex128]:
        pass


class ChannelModel(BaseModel):
    """Class representing a communication channel model."""

    impulse_response: NDArray[np.complex128]  # Channel impulse response in time domain
    noise_model: INoiseModel  # Instance of a noise model implementing INoiseModel
    snr_db: float  # Signal-to-noise ratio in decibels
    channel_type: ChannelType

    model_config = {"arbitrary_types_allowed": True}

    def magnitude_response(self, fft_size: int) -> NDArray[np.complex128]:
        """Compute the channel's magnitude response using FFT."""
        return np.fft.fft(self.impulse_response, n=fft_size)

    def transmit(self, signal: NDArray[np.complex128]) -> NDArray[np.complex128]:
        """
        Transmit a signal through the channel:
        - Serialize blocks into a continuous stream.
        - Apply convolution with the channel.
        - Normalize.
        - Add noise.
        - Deserialize with overlap (IBI preserved).
        """

        h = self.impulse_response
        L = len(h)

        if signal.ndim == 1:
            # Single-carrier case
            serialized = signal
            num_blocks, block_len = 1, len(signal)
        elif signal.ndim == 2:
            num_blocks, block_len = signal.shape
            serialized = signal.reshape(-1)
        else:
            raise ValueError(f"Signal must be 1D or 2D, got {signal.ndim}D")

        # Normalization
        power = np.mean(np.abs(h) ** 2)
        print("Channel impulse response power: ", power)
        if power == 0:
            raise ValueError("Channel impulse response has zero power.")
        h = h / np.sqrt(power)
        print("Normalized channel impulse response power: ", np.mean(np.abs(h) ** 2))

        # Convolution
        filtered_serialized = np.convolve(serialized, h, mode="full")

        # Remove extra samples added by 'full' convolution
        filtered_serialized = filtered_serialized[: len(serialized)]

        # Normalize received power before noise (important for correct SNR)
        rx_power = np.mean(np.abs(filtered_serialized) ** 2)
        filtered_serialized /= np.sqrt(rx_power)

        # Add noise
        noisy_serialized = self.noise_model.add_noise(filtered_serialized, self.snr_db)  # type: ignore

        # --- Proper deserialization ---
        if num_blocks == 1:
            return noisy_serialized

        noisy_blocks = []
        for i in range(num_blocks):
            start = i * block_len
            end = start + block_len + L - 1
            noisy_blocks.append(noisy_serialized[start:end])
        noisy_signal = np.vstack(noisy_blocks)

        return noisy_signal


class AWGNModel(INoiseModel):
    """Additive White Gaussian Noise (AWGN) model."""

    def add_noise(self, signal: NDArray[complex128], snr_db: float) -> NDArray[complex128]:
        """Add AWGN to the signal based on the specified SNR in dB."""
        signal_power = np.mean(np.abs(signal) ** 2)
        snr_linear = 10 ** (snr_db / 10)
        noise_power = signal_power / snr_linear

        # Generate white Gaussian noise
        noise = np.sqrt(noise_power / 2) * (
            np.random.normal(size=signal.shape) + 1j * np.random.normal(size=signal.shape)
        )

        return signal + noise


class NoNoiseModel(INoiseModel):
    """No noise model (passes the signal unchanged)."""

    def add_noise(self, signal: NDArray[complex128], snr_db: float) -> NDArray[complex128]:
        """Return the signal unchanged (no noise added)."""
        return signal
