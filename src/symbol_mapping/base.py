from abc import ABC, abstractmethod
import numpy as np
from numpy.typing import NDArray


class IConstellationSymbolMapper(ABC):
    """Interface for symbol mappers."""

    constellation: NDArray[np.complex128]

    @abstractmethod
    def encode(self, bits: NDArray[np.int_]) -> NDArray[np.complex128]:
        pass

    @abstractmethod
    def decode(self, symbols: NDArray[np.complex128]) -> NDArray[np.int_]:
        pass

    @abstractmethod
    def symbol_to_binary_word(self, symbol: np.complex128) -> str:
        pass

    @abstractmethod
    def get_constellation_name(self) -> str:
        pass


class QAMConstellationSymbolMapper(IConstellationSymbolMapper):
    """
    Maps bits to QAM constellation symbols and vice versa.

    Supports square QAM constellations with Gray coding.
    Normalizes constellation to unit average energy.

    Attributes:
        order (int): Modulation order (e.g., 4 for 4-QAM, 16 for 16-QAM).
        k (int): Bits per symbol (log2(order)).
        constellation (np.ndarray): Complex constellation points.
        gray_map (dict): Mapping from binary index → Gray-coded index.
        inv_gray_map (dict): Inverse Gray mapping.
    """

    def __init__(self, order: int):
        if int(np.sqrt(order)) ** 2 != order:
            raise ValueError("Order must be a perfect square (e.g., 4, 16, 64).")

        self.order = order
        self.k = int(np.log2(order))
        self.constellation = self._generate_constellation()

        # Gray code mapping
        self.gray_map = {i: i ^ (i >> 1) for i in range(order)}
        self.inv_gray_map = {v: k for k, v in self.gray_map.items()}

    def get_constellation_name(self) -> str:
        return f"{self.order}-QAM"

    def _generate_constellation(self) -> NDArray[np.complex128]:
        """Generate normalized square QAM constellation ordered for Gray coding."""
        m_side = int(np.sqrt(self.order))
        re = np.arange(-m_side + 1, m_side, 2)
        im = np.arange(-m_side + 1, m_side, 2)

        # Generate constellation points in natural binary order first
        points = []
        for j in im[::-1]:  # top to bottom (positive to negative imaginary)
            for r in re:  # left to right (negative to positive real)
                points.append(complex(r, j))

        # Now reorder according to Gray code mapping
        # We want constellation[gray_index] = point_at_binary_position
        const = np.zeros(self.order, dtype=complex)
        for binary_idx in range(self.order):
            gray_idx = binary_idx ^ (binary_idx >> 1)  # Convert binary to Gray
            const[gray_idx] = points[binary_idx]

        const /= np.sqrt((np.abs(const) ** 2).mean())  # unit average power
        return const

    def encode(self, bits: NDArray[np.int_]) -> NDArray[np.complex128]:
        """Map input bits → QAM symbols using Gray coding."""
        if len(bits) % self.k != 0:
            raise ValueError("Number of bits must be a multiple of k")

        # Reshape bits into groups of k
        bit_chunks = bits.reshape(-1, self.k)

        # Convert to integers
        bin_indices = bit_chunks.dot(1 << np.arange(self.k - 1, -1, -1))

        # Apply Gray mapping
        gray_indices = np.vectorize(self.gray_map.get)(bin_indices)

        return self.constellation[gray_indices]

    def decode(self, symbols: NDArray[np.complex128]) -> NDArray[np.int_]:
        """Map received symbols → bit sequence using nearest-neighbor decoding."""
        # Compute distances for all symbols at once
        distances = np.abs(self.constellation[:, np.newaxis] - symbols)
        gray_indices = np.argmin(distances, axis=0)

        # Convert Gray indices back to binary indices
        bin_indices = np.vectorize(self.inv_gray_map.get)(gray_indices)

        # Convert binary indices to bit chunks
        bit_chunks = np.array([list(map(int, bin(idx)[2:].zfill(self.k))) for idx in bin_indices])

        return bit_chunks.flatten()

    def symbol_to_binary_word(self, symbol: np.complex128) -> str:
        """Convert a constellation symbol to its corresponding binary word."""
        distances = np.abs(self.constellation - symbol)
        gray_index = int(np.argmin(distances))
        bin_index = self.inv_gray_map[gray_index]
        return bin(bin_index)[2:].zfill(self.k)


class PSKConstellationSymbolMapper(IConstellationSymbolMapper):
    """
    Maps bits to PSK constellation symbols and vice versa.

    Supports PSK constellations with Gray coding.
    Normalizes constellation to unit average energy.

    Attributes:
        order (int): Modulation order (e.g., 4 for QPSK, 8 for 8-PSK).
        k (int): Bits per symbol (log2(order)).
        constellation (np.ndarray): Complex constellation points.
        gray_map (dict): Mapping from binary index → Gray-coded index.
        inv_gray_map (dict): Inverse Gray mapping.
    """

    def __init__(self, order: int):
        if order < 2 or (order & (order - 1)) != 0:
            raise ValueError("Order must be a power of 2 (e.g., 2, 4, 8, 16).")

        self.order = order
        self.k = int(np.log2(order))
        self.constellation = self._generate_constellation()

        # Gray code mapping
        self.gray_map = {i: i ^ (i >> 1) for i in range(order)}
        self.inv_gray_map = {v: k for k, v in self.gray_map.items()}

    def get_constellation_name(self) -> str:
        return f"{self.order}-PSK"

    def _generate_constellation(self) -> NDArray[np.complex128]:
        """Generate normalized PSK constellation ordered for Gray coding."""
        angles = 2 * np.pi * np.arange(self.order) / self.order
        points = np.exp(1j * angles)

        # Reorder according to Gray code mapping
        const = np.zeros(self.order, dtype=complex)
        for binary_idx in range(self.order):
            gray_idx = binary_idx ^ (binary_idx >> 1)  # Convert binary to Gray
            const[gray_idx] = points[binary_idx]

        return const  # Already unit power

    def encode(self, bits: NDArray[np.int_]) -> NDArray[np.complex128]:
        """Map input bits → PSK symbols using Gray coding."""
        if len(bits) % self.k != 0:
            raise ValueError("Number of bits must be a multiple of k")

        # Reshape bits into groups of k
        bit_chunks = bits.reshape(-1, self.k)

        # Convert to integers
        bin_indices = bit_chunks.dot(1 << np.arange(self.k - 1, -1, -1))
        # Apply Gray mapping
        gray_indices = np.vectorize(self.gray_map.get)(bin_indices)
        return self.constellation[gray_indices]

    def decode(self, symbols: NDArray[np.complex128]) -> NDArray[np.int_]:
        """Map received symbols → bit sequence using nearest-neighbor decoding."""
        bits_out = []

        for symbol in symbols:
            distances = np.abs(self.constellation - symbol)
            gray_index = int(np.argmin(distances))

            # Convert Gray index back to binary index
            bin_index = self.inv_gray_map[gray_index]

            # Convert index to bit chunk
            bit_chunk = list(map(int, bin(bin_index)[2:].zfill(self.k)))
            bits_out.extend(bit_chunk)

        return np.array(bits_out, dtype=int)

    def symbol_to_binary_word(self, symbol: np.complex128) -> str:
        """Convert a constellation symbol to its corresponding binary word."""
        distances = np.abs(self.constellation - symbol)
        gray_index = int(np.argmin(distances))
        bin_index = self.inv_gray_map[gray_index]
        return bin(bin_index)[2:].zfill(self.k)
