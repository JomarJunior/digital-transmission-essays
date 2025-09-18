import os
import matplotlib.pyplot as plt
import numpy as np
from bits_generator.base import BitsGenerator
from channel.enums import ChannelType, NoiseType
from channel.factory import ChannelFactory
from configuration.base import Settings, SimulationSettings
from configuration.enums import EqualizationMethod, PrefixType
from equalization.factory import EqualizationFactory
from modulation.base import (
    CyclicPrefixScheme,
    NoPrefixScheme,
    OFDMModulator,
    SerialParallelConverter,
    ZeroPrefixScheme,
)
from symbol_mapping.factory import SymbolMapperFactory


def main():
    # Load general settings
    settings = Settings.from_json("config/settings.json")

    print("-" * 40, "\n")
    print(settings, "\n")
    print("-" * 40)

    # Load simulation settings
    sim_settings = SimulationSettings.from_json("config/simulation_settings.json")
    print(sim_settings)
    print("-" * 40, "\n")

    # -----------------------------------------------------------------
    # Generate random bits
    # -----------------------------------------------------------------
    bits_generator = BitsGenerator()
    random_bits = np.array([], dtype=int)
    if sim_settings.num_bits:
        random_bits = bits_generator.generate_bits(sim_settings.num_bits)
    if sim_settings.num_symbols:
        bits_per_symbol = int(np.log2(sim_settings.constellation_order))
        total_bits = sim_settings.num_symbols * bits_per_symbol
        random_bits = bits_generator.generate_bits(total_bits)
    print("Random Bits:", random_bits, " len=", len(random_bits))

    snr_index = 6  # Index to select SNR value from the list

    # -----------------------------------------------------------------
    # Create channel model
    # -----------------------------------------------------------------
    channel, channel_time_domain_size = None, None
    if sim_settings.channel_type == ChannelType.CUSTOM:
        channel, channel_time_domain_size = ChannelFactory.create_custom_channel_from_file(
            snr_db=sim_settings.signal_noise_ratios[snr_index],
            noise_type=sim_settings.noise_type,
            filepath=sim_settings.channel_model_path,
        )

    if sim_settings.channel_type == ChannelType.FLAT:
        channel, channel_time_domain_size = ChannelFactory.create_flat_channel(
            snr_db=sim_settings.signal_noise_ratios[snr_index],
            noise_type=sim_settings.noise_type,
        )

    if channel is None or channel_time_domain_size is None:
        raise ValueError(f"Unsupported channel type: {sim_settings.channel_type}")

    # -----------------------------------------------------------------
    # Map bits to QAM symbols
    # -----------------------------------------------------------------
    symbol_mapper = SymbolMapperFactory.create_mapper(
        constellation_type=sim_settings.constellation_type,
        order=sim_settings.constellation_order,
    )
    symbols = symbol_mapper.encode(random_bits)

    # -----------------------------------------------------------------
    # OFDM Modulation with Cyclic Prefix
    # -----------------------------------------------------------------
    print("\n" + "-" * 40)
    print("OFDM Modulation with Cyclic Prefix")
    print("-" * 40 + "\n")

    # Use a reasonable cyclic prefix length
    print(f"Channel time domain length: {channel_time_domain_size}")
    cyclic_prefix_length = int(channel_time_domain_size * sim_settings.prefix_length_ratio)

    prefix_scheme = None
    if sim_settings.prefix_type == PrefixType.CYCLIC:
        prefix_scheme = CyclicPrefixScheme(prefix_length=cyclic_prefix_length)

    if sim_settings.prefix_type == PrefixType.NONE:
        prefix_scheme = NoPrefixScheme()

    if sim_settings.prefix_type == PrefixType.ZERO:
        prefix_scheme = ZeroPrefixScheme(prefix_length=cyclic_prefix_length)

    if prefix_scheme is None:
        raise ValueError(f"Unsupported prefix type: {sim_settings.prefix_type}")

    ofdm_modulator = OFDMModulator(
        num_subcarriers=sim_settings.num_bands,
        prefix_scheme=prefix_scheme,
    )
    print("Total Symbols:", len(symbols))
    symbols_reshaped = SerialParallelConverter.to_parallel(
        data=symbols, num_parallel=ofdm_modulator.num_subcarriers
    )
    ofdm_signal = ofdm_modulator.modulate(symbols_reshaped)

    # -----------------------------------------------------------------
    # Simulate transmission over a channel
    # -----------------------------------------------------------------
    print("\n" + "-" * 40)
    print("Transmission over Channel")
    print("-" * 40 + "\n")

    transmitted_signal = ofdm_signal  # OFDM signal is 2D: (num_symbols, fft_size + prefix_length)
    print(f"Transmitted signal shape: {transmitted_signal.shape}")

    # Use the channel's transmit method to handle the 2D OFDM signal
    received_signal = channel.transmit(transmitted_signal)

    print(f"Simulated Channel Type: {channel.channel_type}")
    print(f"Signal-to-Noise Ratio (SNR): {sim_settings.signal_noise_ratios[snr_index]} dB")
    print(f"Received signal shape: {received_signal.shape}")

    # Demodulate received signal
    received_ofdm_symbols = ofdm_modulator.demodulate(received_signal)
    print(f"Received OFDM symbols shape: {received_ofdm_symbols.shape}")

    # -----------------------------------------------------------------
    # Equalization
    # -----------------------------------------------------------------
    print("\n" + "-" * 40)
    print("Equalization")
    print("-" * 40 + "\n")

    # Where to get the noise variance from?
    # For AWGN channel, it can be derived from SNR and signal power. We calculate it here.
    noise_variance = None
    if (
        sim_settings.noise_type == NoiseType.AWGN
        and sim_settings.equalization_method == EqualizationMethod.MMSE
    ):
        signal_power = np.mean(np.abs(transmitted_signal) ** 2)
        snr_linear = 10 ** (sim_settings.signal_noise_ratios[snr_index] / 10)
        noise_variance = signal_power / snr_linear
        print(f"Calculated noise variance for AWGN: {noise_variance}")

    equalizator = EqualizationFactory.create_equalizator(
        sim_settings.equalization_method,
        channel=channel,
        fft_size=sim_settings.num_bands,
        noise_variance=noise_variance,
    )

    received_ofdm_symbols = equalizator.equalize(received_ofdm_symbols)

    # -------------------------------------------------
    # Demodulation
    # -------------------------------------------------
    received_ofdm_symbols_flat = SerialParallelConverter.to_serial(received_ofdm_symbols)
    demodulated_bits_channel = symbol_mapper.decode(received_ofdm_symbols_flat)

    print(
        "Demodulated Bits from Channel:",
        demodulated_bits_channel,
        " len=",
        len(demodulated_bits_channel),
    )
    padding_channel = len(demodulated_bits_channel) - len(random_bits)
    if padding_channel > 0:
        print(f"Note: {padding_channel} extra bits due to OFDM symbol padding and prefix.")
    print(
        f"""Number of Bit Errors:
        {np.sum(random_bits != demodulated_bits_channel[:len(random_bits)])}"""
    )
    ber = np.sum(random_bits != demodulated_bits_channel[: len(random_bits)]) / len(random_bits)
    print(
        f"""Bit Error Rate (BER):
        {ber:.6f}"""
    )
    print("-" * 40 + "\n")

    # -----------------------------------------------------------------
    # Plot constellation diagram
    # -----------------------------------------------------------------
    plot_title = (
        f"Received {symbol_mapper.get_constellation_name()} Constellation "
        f"over {channel.channel_type} Channel with "
        f"{sim_settings.signal_noise_ratios[snr_index]} dB SNR"
    )
    plot_subtitle = f"BER: {ber:.6f}\nNumber of symbols: {len(symbols):.2e}"
    # Clear plots
    plt.clf()
    plt.cla()

    # Plot received constellation
    plt.figure(figsize=(8, 8))
    plt.scatter(
        received_ofdm_symbols_flat.real,
        received_ofdm_symbols_flat.imag,
        color="red",
        s=2,
        zorder=5,
        alpha=0.3,
    )
    # Plot original constellation points for reference
    plt.scatter(
        symbol_mapper.constellation.real,
        symbol_mapper.constellation.imag,
        color="blue",
        s=100,
        zorder=10,
        label="Original Constellation",
    )
    plt.legend()
    plt.title(plot_title)
    plt.title(plot_subtitle, fontsize=10, loc="left", y=0.92)
    plt.xlabel("In-Phase")
    plt.ylabel("Quadrature")
    plt.axhline(0, color="black", linewidth=0.8)  # Add horizontal axis
    plt.axvline(0, color="black", linewidth=0.8)  # Add vertical axis
    plt.grid(False)  # Remove the inner grid
    plt.axis("equal")
    if settings.debug:
        file_name = (
            plot_title.replace(" ", "_")
            .replace("/", "_")
            .replace(".0", "")
            .replace("-", "_")
            .lower()
        )

        file_path = f"images/{channel.channel_type.lower()}_channel/{file_name}.png"
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        plt.savefig(file_path, dpi=300)
    else:
        plt.show()


if __name__ == "__main__":
    main()
