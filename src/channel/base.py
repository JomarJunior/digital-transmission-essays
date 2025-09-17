from abc import ABC, abstractmethod
import numpy as np
from numpy.typing import NDArray
from numpy import complex128
from pydantic import BaseModel
from channel.enums import NoiseType, ChannelType


class INoiseModel(ABC):
    """Interface for noise models."""

    @abstractmethod
    def add_noise(
        self, signal: NDArray[complex128], snr_db: float
    ) -> NDArray[complex128]:
        pass


class ChannelModel(BaseModel):
    """Class representing a communication channel model."""

    magnitude_response: NDArray[np.complex128]  # Frequency response of the channel
    noise_model: INoiseModel  # Instance of a noise model implementing INoiseModel
    snr_db: float  # Signal-to-noise ratio in decibels
    channel_type: ChannelType

    # Because of NDArray
    model_config = {"arbitrary_types_allowed": True}

    def transmit(self, signal: NDArray[np.complex128]) -> NDArray[np.complex128]:
        """Transmit a signal through the channel, applying the channel effects and adding noise."""

        if signal.ndim == 1:
            # Single signal - apply channel effects in frequency domain
            signal_freq = np.fft.fft(signal, n=self.magnitude_response.size)
            # Ensure magnitude_response matches the shape of signal_freq
            if self.magnitude_response.shape != signal_freq.shape:
                # Pad with zeros
                if self.magnitude_response.size < signal_freq.size:
                    padded_response = np.zeros(signal_freq.shape, dtype=np.complex128)
                    padded_response[: self.magnitude_response.size] = (
                        self.magnitude_response
                    )
                    magnitude_response = padded_response
                else:
                    # Truncate
                    magnitude_response = self.magnitude_response[: signal_freq.size]
            else:
                magnitude_response = self.magnitude_response

            filtered_signal_freq = signal_freq * magnitude_response
            filtered_signal = np.fft.ifft(
                filtered_signal_freq, n=magnitude_response.size
            )

        elif signal.ndim == 2:
            # OFDM signal - apply channel effects to each OFDM symbol (each row)
            num_ofdm_symbols, fft_size = signal.shape

            # For flat channel, extend magnitude response to match FFT size if needed
            if self.magnitude_response.size < fft_size:
                magnitude_response = np.ones(fft_size, dtype=complex)
                # For flat channel, just use unit response
            else:
                magnitude_response = self.magnitude_response[:fft_size]

            # Apply channel to each OFDM symbol
            filtered_signal = np.zeros_like(signal)
            for i in range(num_ofdm_symbols):
                # Convert time domain OFDM symbol to frequency domain
                signal_freq = np.fft.fft(signal[i, :])
                # Apply channel response
                filtered_signal_freq = signal_freq * magnitude_response
                # Convert back to time domain
                filtered_signal[i, :] = np.fft.ifft(filtered_signal_freq)
        else:
            raise ValueError(f"Signal must be 1D or 2D, got {signal.ndim}D")

        # Add noise using the specified noise model
        noisy_signal = self.noise_model.add_noise(filtered_signal, self.snr_db)

        return noisy_signal

        # Add noise using the specified noise model
        noisy_signal = self.noise_model.add_noise(filtered_signal, self.snr_db)

        return noisy_signal

    # """Transmit a signal through the channel, applying the channel effects and adding noise."""
    #     # Handle both 1D and 2D signals (for OFDM)
    #     if signal.ndim == 1:
    #         # Single signal - apply channel effects in frequency domain
    #         signal_freq = np.fft.fft(signal)
    #         if self.magnitude_response.shape != signal_freq.shape:
    #             raise ValueError(
    #                 f"Shape mismatch: signal_freq has shape {signal_freq.shape}, "
    #                 f"but magnitude_response has shape {self.magnitude_response.shape}."
    #             )
    #         filtered_signal_freq = signal_freq * self.magnitude_response
    #         filtered_signal = np.fft.ifft(filtered_signal_freq)
    #     elif signal.ndim == 2:
    #         # OFDM signal - apply channel effects to each OFDM symbol (each row)
    #         num_ofdm_symbols, fft_size = signal.shape
    #         if self.magnitude_response.shape[0] != fft_size:
    #             raise ValueError(
    #                 f"Shape mismatch: OFDM signal has FFT size {fft_size}, "
    #                 f"but magnitude_response has shape {self.magnitude_response.shape}."
    #             )

    #         # Apply channel to each OFDM symbol
    #         filtered_signal = np.zeros_like(signal)
    #         for i in range(num_ofdm_symbols):
    #             # The OFDM signal is already in time domain, but we need to apply frequency domain filtering
    #             # Convert to frequency domain, apply channel, convert back
    #             signal_freq = np.fft.fft(signal[i, :])
    #             filtered_signal_freq = signal_freq * self.magnitude_response
    #             filtered_signal[i, :] = np.fft.ifft(filtered_signal_freq)
    #     else:
    #         raise ValueError(f"Signal must be 1D or 2D, got {signal.ndim}D")

    #     # Add noise using the specified noise model
    #     noisy_signal = self.noise_model.add_noise(filtered_signal, self.snr_db)

    #     return noisy_signal


class AWGNModel(INoiseModel):
    """Additive White Gaussian Noise (AWGN) model."""

    def add_noise(
        self, signal: NDArray[complex128], snr_db: float
    ) -> NDArray[complex128]:
        """Add AWGN to the signal based on the specified SNR in dB."""
        signal_power = np.mean(np.abs(signal) ** 2)
        snr_linear = 10 ** (snr_db / 10)
        noise_power = signal_power / snr_linear

        # Generate white Gaussian noise
        noise = np.sqrt(noise_power / 2) * (
            np.random.normal(size=signal.shape)
            + 1j * np.random.normal(size=signal.shape)
        )

        return signal + noise


class NoNoiseModel(INoiseModel):
    """No noise model (passes the signal unchanged)."""

    def add_noise(
        self, signal: NDArray[complex128], snr_db: float
    ) -> NDArray[complex128]:
        """Return the signal unchanged (no noise added)."""
        return signal
