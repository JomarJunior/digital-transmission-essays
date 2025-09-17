import numpy as np
import matplotlib.pyplot as plt
from configuration.base import Settings, SimulationSettings
from bits_generator.base import BitsGenerator
from symbol_mapping.factory import SymbolMapperFactory
from modulation.base import CyclicPrefixScheme, OFDMModulator, SerialParallelConverter
from channel.factory import ChannelFactory
from channel.enums import NoiseType


def test_ofdm_full():
    # Load settings
    sim_settings = SimulationSettings.from_json("config/simulation_settings.json")

    # Use moderate number of bits for testing
    num_test_bits = 16000  # Should give nice OFDM symbols
    bits_generator = BitsGenerator()
    random_bits = bits_generator.generate_bits(num_test_bits)
    print(f"Testing with {num_test_bits} bits")

    # Create symbol mapper
    symbol_mapper = SymbolMapperFactory.create_mapper(
        constellation_type=sim_settings.constellation_type,
        order=sim_settings.constellation_order,
    )
    symbols = symbol_mapper.encode(random_bits)
    print(f"Generated {len(symbols)} symbols")

    # Create channel with no noise
    channel = ChannelFactory.create_flat_channel(
        snr_db=100.0, noise_type=NoiseType.NONE  # Very high SNR, no noise
    )

    # OFDM Modulation with reasonable cyclic prefix
    cyclic_prefix_length = sim_settings.num_bands // 4  # 16 for 64 subcarriers

    ofdm_modulator = OFDMModulator(
        num_subcarriers=sim_settings.num_bands,
        prefix_scheme=CyclicPrefixScheme(prefix_length=cyclic_prefix_length),
    )

    # Convert to parallel and modulate
    symbols_reshaped = SerialParallelConverter.to_parallel(
        data=symbols, num_parallel=ofdm_modulator.num_subcarriers
    )
    print(f"OFDM symbols shape: {symbols_reshaped.shape}")

    ofdm_signal = ofdm_modulator.modulate(symbols_reshaped)
    print(f"OFDM signal shape: {ofdm_signal.shape}")

    # Transmit through channel (no noise, flat channel)
    received_signal = channel.transmit(ofdm_signal)
    print(f"Received signal shape: {received_signal.shape}")

    # OFDM demodulate
    received_ofdm_symbols = ofdm_modulator.demodulate(received_signal)
    received_symbols_flat = SerialParallelConverter.to_serial(received_ofdm_symbols)

    # Decode symbols
    demodulated_bits = symbol_mapper.decode(received_symbols_flat)

    # Compare results
    errors = np.sum(random_bits != demodulated_bits[: len(random_bits)])
    print(f"Bit errors: {errors} out of {len(random_bits)}")
    print(f"BER: {errors / len(random_bits):.6f}")

    # Test with real noise (20 dB SNR)
    channel_with_noise = ChannelFactory.create_flat_channel(
        snr_db=20.0, noise_type=NoiseType.AWGN
    )

    received_signal_noisy = channel_with_noise.transmit(ofdm_signal)
    received_ofdm_symbols_noisy = ofdm_modulator.demodulate(received_signal_noisy)
    received_symbols_flat_noisy = SerialParallelConverter.to_serial(
        received_ofdm_symbols_noisy
    )
    demodulated_bits_noisy = symbol_mapper.decode(received_symbols_flat_noisy)

    errors_noisy = np.sum(random_bits != demodulated_bits_noisy[: len(random_bits)])
    print(f"With 20dB AWGN - Bit errors: {errors_noisy} out of {len(random_bits)}")
    print(f"With 20dB AWGN - BER: {errors_noisy / len(random_bits):.6f}")


if __name__ == "__main__":
    test_ofdm_full()
