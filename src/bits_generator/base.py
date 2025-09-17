from typing import Optional
from numpy.random import Generator, PCG64
from numpy import ndarray


class BitsGenerator:
    def __init__(self, generator: Optional[Generator] = None):
        self.generator = generator or Generator(PCG64())

    def generate_bits(self, num_bits: int) -> ndarray:
        """Generate a random array of bits (0s and 1s)."""
        if num_bits <= 0:
            raise ValueError("Number of bits must be a positive integer.")
        return self.generator.integers(0, 2, num_bits)
