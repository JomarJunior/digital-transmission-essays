import numpy as np
from src.channel.base import ChannelModel, AWGNModel

# Test the fixed channel implementation
print("Testing Channel with different signal shapes...")

# Create a simple channel model
magnitude_response = np.ones(128, dtype=complex)  # Flat channel
noise_model = AWGNModel()
channel = ChannelModel(
    magnitude_response=magnitude_response, noise_model=noise_model, snr_db=20.0
)

# Test 1D signal
print("\n1D Signal Test:")
signal_1d = np.random.random(128) + 1j * np.random.random(128)
print(f"Input 1D signal shape: {signal_1d.shape}")
try:
    output_1d = channel.transmit(signal_1d)
    print(f"Output 1D signal shape: {output_1d.shape}")
    print("✓ 1D signal transmission successful")
except Exception as e:
    print(f"✗ 1D signal transmission failed: {e}")

# Test 2D signal (OFDM-like)
print("\n2D Signal Test:")
signal_2d = np.random.random((1000, 128)) + 1j * np.random.random((1000, 128))
print(f"Input 2D signal shape: {signal_2d.shape}")
try:
    output_2d = channel.transmit(signal_2d)
    print(f"Output 2D signal shape: {output_2d.shape}")
    print("✓ 2D signal transmission successful")
except Exception as e:
    print(f"✗ 2D signal transmission failed: {e}")

# Test incompatible shapes
print("\nIncompatible Shape Test:")
signal_wrong = np.random.random((1000, 64)) + 1j * np.random.random((1000, 64))
print(f"Input wrong signal shape: {signal_wrong.shape}")
try:
    output_wrong = channel.transmit(signal_wrong)
    print(f"Output wrong signal shape: {output_wrong.shape}")
    print("✗ Should have failed but didn't")
except Exception as e:
    print(f"✓ Correctly caught incompatible shape: {e}")
