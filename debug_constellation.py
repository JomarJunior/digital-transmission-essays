import numpy as np
import matplotlib.pyplot as plt
from src.symbol_mapping.base import QAMConstellationSymbolMapper

# Create a 4-QAM mapper
qam_mapper = QAMConstellationSymbolMapper(order=4)

print("Constellation points:")
for i, symbol in enumerate(qam_mapper.constellation):
    print(f"Index {i}: {symbol:.6f}")

print("\nGray map (binary index -> Gray index):")
for k, v in qam_mapper.gray_map.items():
    print(f"{k:02b} -> {v:02b} (Gray)")

print("\nInverse Gray map (Gray index -> binary index):")
for k, v in qam_mapper.inv_gray_map.items():
    print(f"{k:02b} (Gray) -> {v:02b}")

print("\nSymbol to binary word mapping:")
for i, symbol in enumerate(qam_mapper.constellation):
    word = qam_mapper.symbol_to_binary_word(symbol)
    print(f"Constellation[{i}] = {symbol:.6f} -> '{word}'")

# Let's also check what happens during encoding
print("\nEncoding test:")
test_bits = np.array([0, 0, 0, 1, 1, 0, 1, 1])  # 00, 01, 10, 11
symbols = qam_mapper.encode(test_bits)
for i in range(0, len(test_bits), 2):
    bit_pair = test_bits[i : i + 2]
    symbol = symbols[i // 2]
    print(f"Bits {bit_pair} -> Symbol {symbol:.6f}")
