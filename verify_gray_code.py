import numpy as np
import matplotlib.pyplot as plt
from src.symbol_mapping.base import QAMConstellationSymbolMapper

# Create a 4-QAM mapper
qam_mapper = QAMConstellationSymbolMapper(order=4)

# Create a simple plot to verify Gray code layout
plt.figure(figsize=(8, 6))
plt.scatter(
    qam_mapper.constellation.real,
    qam_mapper.constellation.imag,
    color="blue",
    s=100,
    zorder=5,
)

# Display the constellation symbols words
for symbol in qam_mapper.constellation:
    word = qam_mapper.symbol_to_binary_word(symbol)
    plt.text(
        symbol.real,
        symbol.imag + 0.15,
        word,
        fontsize=16,
        ha="center",
        va="center",
        color="white",
        bbox={"facecolor": "black", "alpha": 0.7, "boxstyle": "round,pad=0.3"},
    )

plt.title("4-QAM Constellation with Gray Code Labels", fontsize=14)
plt.xlabel("In-Phase", fontsize=12)
plt.ylabel("Quadrature", fontsize=12)
plt.axhline(0, color="black", linewidth=0.8, alpha=0.5)
plt.axvline(0, color="black", linewidth=0.8, alpha=0.5)
plt.grid(True, alpha=0.3)
plt.axis("equal")

# Add annotations to show the layout
plt.text(-0.5, 0.9, "00", fontsize=14, ha="center", weight="bold", color="red")
plt.text(0.5, 0.9, "01", fontsize=14, ha="center", weight="bold", color="red")
plt.text(-0.5, -0.9, "10", fontsize=14, ha="center", weight="bold", color="red")
plt.text(0.5, -0.9, "11", fontsize=14, ha="center", weight="bold", color="red")

plt.tight_layout()
plt.savefig("gray_code_verification.png", dpi=150, bbox_inches="tight")
plt.show()

print("Constellation layout verification:")
print("Current constellation points and their labels:")
for i, symbol in enumerate(qam_mapper.constellation):
    word = qam_mapper.symbol_to_binary_word(symbol)
    quadrant = ""
    if symbol.real > 0 and symbol.imag > 0:
        quadrant = "Top-Right"
    elif symbol.real < 0 and symbol.imag > 0:
        quadrant = "Top-Left"
    elif symbol.real < 0 and symbol.imag < 0:
        quadrant = "Bottom-Left"
    else:
        quadrant = "Bottom-Right"

    print(f"  {word}: {symbol:.3f} ({quadrant})")

print("\nFor proper Gray coding, adjacent symbols should differ by only 1 bit:")
print("- Horizontally adjacent: 00↔01 and 10↔11 ✓")
print("- Vertically adjacent: 00↔10 and 01↔11 ✓")
