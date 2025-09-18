from abc import ABC, abstractmethod
from numpy.typing import NDArray
from numpy import complex128
import numpy as np


class IModulator(ABC):
    """Interface for modulation schemes."""

    @abstractmethod
    def modulate(self, symbols: NDArray[complex128]) -> NDArray[complex128]:
        pass

    @abstractmethod
    def demodulate(self, received_signal: NDArray[complex128]) -> NDArray[complex128]:
        pass


class IPrefixScheme(ABC):
    """Interface for prefix schemes."""

    @abstractmethod
    def add_prefix(self, symbols: NDArray[complex128]) -> NDArray[complex128]:
        pass

    @abstractmethod
    def remove_prefix(self, received_signal: NDArray[complex128]) -> NDArray[complex128]:
        pass


class CyclicPrefixScheme(IPrefixScheme):
    """Implements cyclic prefix addition and removal."""

    def __init__(self, prefix_length: int):
        self.prefix_length = prefix_length

    def add_prefix(self, symbols: NDArray[complex128]) -> NDArray[complex128]:
        """Add cyclic prefix to symbols (works for 1D or 2D arrays)."""
        length = self.prefix_length

        if symbols.ndim == 1:
            if length > symbols.size:
                raise ValueError("Prefix length longer than symbol length.")
            prefix = symbols[-length:]
            return np.concatenate((prefix, symbols), axis=0)

        if symbols.ndim == 2:
            _, symbols_size = symbols.shape
            if length > symbols_size:
                raise ValueError("Prefix length longer than symbol length.")
            prefix = symbols[:, -length:]
            return np.concatenate((prefix, symbols), axis=1)

        raise ValueError("Input symbols must be 1D or 2D array.")

    def remove_prefix(self, received_signal: NDArray[complex128]) -> NDArray[complex128]:
        """Remove cyclic prefix from received signal."""
        if received_signal.ndim == 1:
            return received_signal[self.prefix_length :]
        if received_signal.ndim == 2:
            return received_signal[:, self.prefix_length :]

        raise ValueError("Input received_signal must be 1D or 2D array.")


class NoPrefixScheme(IPrefixScheme):
    """No prefix scheme (pass-through)."""

    def add_prefix(self, symbols: NDArray[complex128]) -> NDArray[complex128]:
        """No prefix added."""
        return symbols

    def remove_prefix(self, received_signal: NDArray[complex128]) -> NDArray[complex128]:
        """No prefix to remove."""
        return received_signal


class ZeroPrefixScheme(IPrefixScheme):
    """Implements zero prefix addition and removal."""

    def __init__(self, prefix_length: int):
        self.prefix_length = prefix_length

    def add_prefix(self, symbols: NDArray[complex128]) -> NDArray[complex128]:
        """Add zero prefix to symbols."""
        if symbols.ndim == 1:
            prefix = np.zeros(self.prefix_length, dtype=symbols.dtype)
            return np.concatenate((prefix, symbols))
        if symbols.ndim == 2:
            prefix = np.zeros((symbols.shape[0], self.prefix_length), dtype=symbols.dtype)
            return np.hstack((prefix, symbols))

        raise ValueError("Input symbols must be 1D or 2D array.")

    def remove_prefix(self, received_signal: NDArray[complex128]) -> NDArray[complex128]:
        """Remove zero prefix from received signal."""
        if received_signal.ndim == 1:
            return received_signal[self.prefix_length :]
        if received_signal.ndim == 2:
            return received_signal[:, self.prefix_length :]

        raise ValueError("Input received_signal must be 1D or 2D array.")


class OFDMModulator(IModulator):
    """Implements OFDM modulation and demodulation."""

    def __init__(self, num_subcarriers: int, prefix_scheme: IPrefixScheme):
        self.num_subcarriers = num_subcarriers
        self.prefix_scheme = prefix_scheme

    def modulate(self, symbols: NDArray[complex128]) -> NDArray[complex128]:
        """OFDM modulation with IFFT and prefix addition."""
        if symbols.shape[1] != self.num_subcarriers:
            raise ValueError("Number of symbols must match number of subcarriers.")

        # Perform IFFT
        time_domain_signal = np.fft.ifft(symbols, axis=1, norm="ortho", n=self.num_subcarriers)

        # Add prefix
        return self.prefix_scheme.add_prefix(time_domain_signal)

    def demodulate(self, received_signal: NDArray[complex128]) -> NDArray[complex128]:
        """OFDM demodulation with prefix removal and FFT."""
        # Remove prefix
        signal_no_prefix = self.prefix_scheme.remove_prefix(received_signal)

        # Perform FFT
        return np.fft.fft(signal_no_prefix, axis=1, norm="ortho", n=self.num_subcarriers)


class SerialParallelConverter:
    """Utility class for serial-to-parallel and parallel-to-serial conversion."""

    @staticmethod
    def to_parallel(data: NDArray[complex128], num_parallel: int) -> NDArray[complex128]:
        """Convert serial data to parallel format."""
        if len(data) % num_parallel != 0:
            print("Padding data for parallel conversion.")
            # If data length is not a multiple of num_parallel, pad with zeros
            padding_length = num_parallel - (len(data) % num_parallel)
            data = np.concatenate((data, np.zeros(padding_length, dtype=data.dtype)))
        return data.reshape(-1, num_parallel, order="C")  # Column-major order

    @staticmethod
    def to_serial(data: NDArray[complex128]) -> NDArray[complex128]:
        """Convert parallel data back to serial format."""
        return data.flatten(order="C")  # Column-major order
