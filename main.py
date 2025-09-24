import os
from typing import Optional
import numpy as np
from PIL import Image
from configuration.base import Settings, SimulationSettings
from configuration.enums import PrefixType
from equalization.factory import EqualizationFactory
from modulation.base import (
    CyclicPrefixScheme,
    NoPrefixScheme,
    OFDMModulator,
    SerialParallelConverter,
    ZeroPrefixScheme,
)
from simulation.base import CalculateBERStep, PlotConstellationStep, Simulation
from bits_generator.base import BitsGenerator
from channel.factory import ChannelFactory, ChannelType
from symbol_mapping.factory import SymbolMapperFactory

if __name__ == "__main__":
    # Load general settings
    settings = Settings.from_json("config/settings.json")

    print("-" * 40, "\n")
    print(settings, "\n")
    print("-" * 40)

    # Load simulation settings
    sim_settings = SimulationSettings.from_json("config/simulation_settings.json")
    print(sim_settings)
    print("-" * 40, "\n")

    # Create Bits Generator
    bits_generator = BitsGenerator()
    print(f"Bits Generator: {type(bits_generator).__name__}\n")
    print("-" * 40)

    # Create Channel Models
    channels, channel_time_domain_sizes = [], []
    for snr_db in sim_settings.signal_noise_ratios:
        channel, channel_time_domain_size = None, None
        if sim_settings.channel_type == ChannelType.CUSTOM:
            channel, channel_time_domain_size = ChannelFactory.create_custom_channel_from_file(
                snr_db=snr_db,
                noise_type=sim_settings.noise_type,
                filepath=sim_settings.channel_model_path,
            )

        if sim_settings.channel_type == ChannelType.FLAT:
            channel, channel_time_domain_size = ChannelFactory.create_flat_channel(
                snr_db=snr_db,
                noise_type=sim_settings.noise_type,
            )

        if channel is None or channel_time_domain_size is None:
            raise ValueError(f"Unsupported channel type: {sim_settings.channel_type}")

        channels.append(channel)
        channel_time_domain_sizes.append(channel_time_domain_size)
        print(f"Channel Model (SNR={snr_db} dB): {type(channel).__name__}")
    print("-" * 40)

    # Create Symbol Mapper
    symbol_mapper = SymbolMapperFactory.create_mapper(
        constellation_type=sim_settings.constellation_type,
        order=sim_settings.constellation_order,
    )
    print(f"Symbol Mapper: {type(symbol_mapper).__name__}")
    print("-" * 40)

    # Create Modulator
    prefix_lengths = np.array(channel_time_domain_sizes) * sim_settings.prefix_length_ratio
    PREFIX_SCHEME_MAP = {
        PrefixType.CYCLIC: lambda pl: CyclicPrefixScheme(prefix_length=pl),
        PrefixType.ZERO: lambda pl: ZeroPrefixScheme(prefix_length=pl),
        PrefixType.NONE: lambda pl: NoPrefixScheme(),
    }
    if sim_settings.prefix_type not in PREFIX_SCHEME_MAP:
        raise ValueError(f"Unsupported prefix type: {sim_settings.prefix_type}")

    prefix_schemes = [PREFIX_SCHEME_MAP[sim_settings.prefix_type](int(pl)) for pl in prefix_lengths]

    modulators = []
    for i, prefix_scheme in enumerate(prefix_schemes):
        modulators.append(
            OFDMModulator(
                num_subcarriers=sim_settings.num_bands,
                prefix_scheme=prefix_scheme,
            )
        )

    for i, modulator in enumerate(modulators):
        print(
            f"OFDM Modulator {i+1}: {type(modulator).__name__} with {type(modulator.prefix_scheme).__name__}"  # pylint: disable=line-too-long
        )
    print("-" * 40)

    # Create SerialParallelConverter
    serial_parallel_converter = SerialParallelConverter()

    # Create Equalizators
    equalizators = []
    for channel in channels:
        equalizator = EqualizationFactory.create_equalizator(
            sim_settings.equalization_method,
            channel=channel,
            fft_size=sim_settings.num_bands,
        )
        equalizators.append(equalizator)

    # Create Simulations
    simulations = []
    for i, _ in enumerate(sim_settings.signal_noise_ratios):
        simulations.append(
            Simulation(
                configuration=sim_settings,
                bits_generator=bits_generator,
                symbol_mapper=symbol_mapper,
                serial_parallel_converter=serial_parallel_converter,
                channel=channels[i],
                modulator=modulators[i],
                equalizator=equalizators[i],
            )
        )

    print(f"Created {len(simulations)} simulations.")
    # Run Simulations
    for i, simulation in enumerate(simulations):
        print("-" * 40)
        print(f"\nRunning Simulation {i+1} with SNR={sim_settings.signal_noise_ratios[i]} dB")
        results = simulation.run()
        ber = results.get(CalculateBERStep.__output_key__, None)
        if ber is not None:
            print(f"Simulation {i+1} BER: {ber:.6f}")
        else:
            print(f"Simulation {i+1} BER not available.")
        print("-" * 40)

        calculated_snr = results.get("snr", None)
        if calculated_snr is not None:
            print(f"Calculated SNR: {calculated_snr:.2f} dB")
        else:
            print("Calculated SNR not available.")

        # Save plots if available
        constellation_plot: Optional[Image.Image] = results.get(
            PlotConstellationStep.__output_key__, None
        )
        if constellation_plot is not None:
            if not isinstance(constellation_plot, Image.Image):
                print("Constellation plot is not a valid Image object.")
                continue
            full_path = (
                results.get("metadata", {})
                .get("images", {})
                .get("constellation_diagram_full_path", "images/constellation_diagram.png")
            )
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            constellation_plot.save(full_path)
            print(f"Constellation diagram saved to {full_path}")
        print("\n" + "-" * 40 + "\n")
