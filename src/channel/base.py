from abc import ABC, abstractmethod
import numpy as np
from numpy.typing import NDArray
from numpy import complex128
from pydantic import BaseModel
from channel.enums import NoiseType, ChannelType


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
        - Applies linear convolution with the channel impulse response.
        - Adds noise afterward.
        The channel is agnostic to prefixing (CP, ZP, or none).
        """

        h = self.impulse_response

        if signal.ndim == 1:
            # Single-carrier case
            filtered_signal = np.convolve(signal, h, mode="full")

        elif signal.ndim == 2:
            # OFDM case: apply convolution per OFDM symbol (row)
            num_symbols, n = signal.shape
            filtered_signal = np.zeros((num_symbols, n + len(h) - 1), dtype=np.complex128)

            for i in range(num_symbols):
                filtered_signal[i, :] = np.convolve(signal[i, :], h, mode="full")
        else:
            raise ValueError(f"Signal must be 1D or 2D, got {signal.ndim}D")

        # Normalize channel energy to 1
        # Ensures SNR definition is consistent
        energy = np.sum(np.abs(h) ** 2)
        if energy > 0:
            filtered_signal /= np.sqrt(energy)

        # Add noise using the specified noise model
        noisy_signal = self.noise_model.add_noise(filtered_signal, self.snr_db)  # type: ignore

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
