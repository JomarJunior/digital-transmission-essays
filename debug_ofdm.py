import numpy as np
import matplotlib.pyplot as plt
from configuration.base import Settings, SimulationSettings
from bits_generator.base import BitsGenerator
from symbol_mapping.factory import SymbolMapperFactory
from modulation.base import CyclicPrefixScheme, OFDMModulator, SerialParallelConverter
from channel.factory import ChannelFactory
from channel.enums import NoiseType


def debug_ofdm():
    # Load settings
    sim_settings = SimulationSettings.from_json("config/simulation_settings.json")

    # Use small number of bits for debugging
    num_debug_bits = (
        128  # 16-PSK needs 4 bits per symbol -> 32 symbols -> 0.5 OFDM symbols
    )
    bits_generator = BitsGenerator()
    random_bits = bits_generator.generate_bits(num_debug_bits)
    print("Original bits:", random_bits)

    # Create symbol mapper
    symbol_mapper = SymbolMapperFactory.create_mapper(
        constellation_type=sim_settings.constellation_type,
        order=sim_settings.constellation_order,
    )
    symbols = symbol_mapper.encode(random_bits)
    print(f"Symbols shape: {symbols.shape}")
    print("Original symbols:", symbols[:8])  # Show first 8 symbols

    # Create channel (no noise for debugging)
    channel = ChannelFactory.create_flat_channel(
        snr_db=100.0, noise_type=NoiseType.NONE  # Very high SNR, no noise
    )

    # OFDM Modulation
    ofdm_modulator = OFDMModulator(
        num_subcarriers=sim_settings.num_bands,
        prefix_scheme=CyclicPrefixScheme(prefix_length=16),  # Small CP for debugging
    )

    print(f"Number of symbols: {len(symbols)}")
    print(f"Number of subcarriers: {sim_settings.num_bands}")

    # Convert to parallel
    symbols_reshaped = SerialParallelConverter.to_parallel(
        data=symbols, num_parallel=ofdm_modulator.num_subcarriers
    )
    print(f"Symbols reshaped shape: {symbols_reshaped.shape}")
    print("First OFDM symbol (frequency domain):", symbols_reshaped[0])

    # OFDM modulate
    ofdm_signal = ofdm_modulator.modulate(symbols_reshaped)
    print(f"OFDM signal shape: {ofdm_signal.shape}")
    print("First OFDM signal (time domain):", ofdm_signal[0])

    # Transmit through channel (no noise, flat channel)
    received_signal = channel.transmit(ofdm_signal)
    print(f"Received signal shape: {received_signal.shape}")
    print("Received signal (time domain):", received_signal[0])

    # OFDM demodulate
    received_ofdm_symbols = ofdm_modulator.demodulate(received_signal)
    print(f"Received OFDM symbols shape: {received_ofdm_symbols.shape}")
    print("First received OFDM symbol (frequency domain):", received_ofdm_symbols[0])

    # Convert back to serial
    received_symbols_flat = SerialParallelConverter.to_serial(received_ofdm_symbols)
    print(f"Received symbols flat shape: {received_symbols_flat.shape}")
    print("Received symbols:", received_symbols_flat[:8])  # Compare with original

    # Decode symbols
    demodulated_bits = symbol_mapper.decode(received_symbols_flat)
    print("Demodulated bits:", demodulated_bits[: len(random_bits)])

    # Compare
    errors = np.sum(random_bits != demodulated_bits[: len(random_bits)])
    print(f"Bit errors: {errors} out of {len(random_bits)}")
    print(f"BER: {errors / len(random_bits):.6f}")

    # Check if symbols are close to original
    symbol_errors = np.sum(
        np.abs(symbols - received_symbols_flat[: len(symbols)]) > 0.1
    )
    print(f"Symbol errors (>0.1 distance): {symbol_errors} out of {len(symbols)}")


if __name__ == "__main__":
    debug_ofdm()
