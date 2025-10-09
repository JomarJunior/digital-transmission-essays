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
        """Add zero suffix to symbols."""
        print(f"Adding zero prefix of length {self.prefix_length}")
        if symbols.ndim == 1:
            prefix = np.zeros(self.prefix_length, dtype=symbols.dtype)
            return np.concatenate((symbols, prefix))
        if symbols.ndim == 2:
            prefix = np.zeros((symbols.shape[0], self.prefix_length), dtype=symbols.dtype)
            return np.hstack((symbols, prefix))

        raise ValueError("Input symbols must be 1D or 2D array.")

    def remove_prefix(self, received_signal: NDArray[complex128]) -> NDArray[complex128]:
        """Remove zero prefix from received signal."""
        ## @todo Add overlap-and-add
        if received_signal.ndim == 1:
            return self._overlap_and_add(received_signal)
        if received_signal.ndim == 2:
            return np.array([self._overlap_and_add(row) for row in received_signal])

        raise ValueError("Input received_signal must be 1D or 2D array.")

    def _overlap_and_add(self, received_signal: NDArray[complex128]) -> NDArray[complex128]:
        length: int = len(received_signal)
        # Create the identity matrix
        identity: NDArray[complex128] = np.identity(length - self.prefix_length, complex128)

        # Create the overlap and add matrix with zeros
        combined_matrix: NDArray[complex128] = identity.copy()
        for i in range(self.prefix_length):
            overlap_vector = np.zeros(length - self.prefix_length, dtype=complex128)
            overlap_vector[i] = 1.0
            # Concatenate the overlap vector to the combined matrix creating a new column
            combined_matrix = np.hstack([combined_matrix, overlap_vector.reshape(-1, 1)])

        # Perform matrix multiplication to achieve overlap and add
        return combined_matrix @ received_signal


class OFDMModulator(IModulator):
    """Implements OFDM modulation and demodulation."""

    def __init__(self, num_subcarriers: int, prefix_scheme: IPrefixScheme):
        self.num_subcarriers = num_subcarriers
        self.prefix_scheme = prefix_scheme

    def modulate(self, symbols: NDArray[complex128]) -> NDArray[complex128]:
        """OFDM modulation with IFFT and prefix addition."""
        if symbols.shape[1] != self.num_subcarriers:
            raise ValueError("Number of symbols must match number of subcarriers.")

        print(f"Symbols shape (before IFFT): {symbols.shape}")

        # Perform IFFT
        time_domain_signal = np.fft.ifft(symbols, axis=1, norm="ortho", n=self.num_subcarriers)

        print(f"Time domain signal shape (after IFFT): {time_domain_signal.shape}")

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
