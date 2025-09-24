from abc import ABC, abstractmethod
from typing import Optional
import numpy as np
from numpy.typing import NDArray
from numpy import complex128
from channel.base import ChannelModel


class IEqualizator(ABC):
    """
    Interface for equalizators in communication systems.
    """

    @abstractmethod
    def equalize(
        self, signal: NDArray[complex128], noise_variance: Optional[float]
    ) -> NDArray[complex128]:
        pass


class OFDMZeroForcingEqualizator(IEqualizator):
    """
    Zero-Forcing Equalizator for OFDM systems.
    """

    def __init__(self, channel: ChannelModel, fft_size: int):
        self.channel = channel
        self.fft_size = fft_size
        self.h_freq = self.channel.magnitude_response(fft_size)

    def equalize(
        self, signal: NDArray[complex128], noise_variance: Optional[float]
    ) -> NDArray[complex128]:
        """
        Equalize the received OFDM symbols using Zero-Forcing method.
        """
        if signal.ndim != 2:
            raise ValueError("Signal must be a 2D array of shape (num_symbols, fft_size)")

        # Avoid division by zero
        h_freq_safe = np.where(np.abs(self.h_freq) < 1e-10, 1e-10, self.h_freq)

        # Perform equalization in the frequency domain
        equalized_signal = signal / h_freq_safe

        return equalized_signal


class OFDMMMSEEqualizator(IEqualizator):
    """
    MMSE Equalizator for OFDM systems.
    """

    def __init__(self, channel: ChannelModel, fft_size: int):
        self.channel = channel
        self.fft_size = fft_size
        self.h_freq = self.channel.magnitude_response(fft_size)

    def equalize(
        self, signal: NDArray[complex128], noise_variance: Optional[float]
    ) -> NDArray[complex128]:
        """
        Equalize the received OFDM symbols using MMSE method.
        """
        if signal.ndim != 2:
            raise ValueError("Signal must be a 2D array of shape (num_symbols, fft_size)")

        if noise_variance is None:
            raise ValueError("Noise variance must be provided for MMSE equalization.")

        # Compute the MMSE equalization factor
        h_conj = np.conj(self.h_freq)
        h_mag_sq = np.abs(self.h_freq) ** 2
        mmse_factor = h_conj / (h_mag_sq + noise_variance)

        # Perform equalization in the frequency domain
        equalized_signal = signal * mmse_factor

        return equalized_signal


class NoEqualizator(IEqualizator):
    """
    No Equalization, returns the signal as is.
    """

    def __init__(self, channel=None, fft_size=None):
        pass

    def equalize(
        self, signal: NDArray[complex128], noise_variance: Optional[float]
    ) -> NDArray[complex128]:
        return signal
