"""
For this module, we are going to follow the Chain of Responsibility pattern.
Each simulation step will be handled by a specific class, and they will be chained together.
"""

from io import BytesIO
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from bits_generator.base import BitsGenerator
from symbol_mapping.base import IConstellationSymbolMapper
from modulation.base import IModulator, SerialParallelConverter
from channel.base import ChannelModel
from equalization.base import IEqualizator
from configuration.base import SimulationSettings

ChainDataType = Dict[str, Any]


class ISimulationStep(ABC):
    @abstractmethod
    def set_next(self, step: "ISimulationStep") -> "ISimulationStep":
        pass

    @abstractmethod
    def execute(self, data: dict) -> dict:
        pass


class BaseSimulationStep(ISimulationStep):
    _next_step: Optional[ISimulationStep] = None
    __output_key__: str = ""

    def set_next(self, step: ISimulationStep) -> ISimulationStep:
        self._next_step = step
        return step

    def execute(self, data: dict) -> dict:
        if self._next_step:
            return self._next_step.execute(data)
        return data


class SignalGenerationStep(BaseSimulationStep):
    __output_key__ = "input_bits"

    def __init__(
        self,
        bits_generator: BitsGenerator,
        num_bits: Optional[int] = None,
        num_symbols: Optional[int] = None,
        constellation_order: Optional[int] = None,
    ):
        if not num_bits and not num_symbols:
            raise ValueError("Either num_bits or num_symbols must be provided.")
        if num_bits and num_symbols:
            raise ValueError("Only one of num_bits or num_symbols should be provided.")
        if num_symbols and not constellation_order:
            raise ValueError("constellation_order must be provided when num_symbols is specified.")
        self.bits_generator = bits_generator
        self.num_bits = num_bits
        self.num_symbols = num_symbols
        self.constellation_order = constellation_order

    def execute(self, data: ChainDataType) -> ChainDataType:
        """
        Generate random bits based on the provided configuration.
        """
        print("-" * 40)
        print("Starting bit generation...")
        if self.num_bits:
            print("Received num_bits: ", self.num_bits)
            print("Generating bits...")
            # If num_bits is specified, generate that many bits
            bits = self.bits_generator.generate_bits(self.num_bits)
        elif self.num_symbols and self.constellation_order:
            print("Received num_symbols: ", self.num_symbols)
            print("Received constellation_order: ", self.constellation_order)
            print("Calculating total bits to generate...")
            # If num_symbols is specified, calculate the required number of bits
            bits_per_symbol = int(np.log2(self.constellation_order))
            total_bits = self.num_symbols * bits_per_symbol
            bits = self.bits_generator.generate_bits(total_bits)
        else:
            # This should never happen due to checks in __init__
            raise ValueError("Invalid configuration for signal generation.")

        print("Generated bits: ", bits, f" (Total: {len(bits)} bits)")
        data[self.__output_key__] = bits
        return super().execute(data)


class ConstellationMappingStep(BaseSimulationStep):
    __output_key__ = "constellation_symbols"

    def __init__(self, symbol_mapper: IConstellationSymbolMapper):
        self.symbol_mapper = symbol_mapper

    def execute(self, data: ChainDataType) -> ChainDataType:
        """
        Map input bits to constellation symbols.
        """
        print("-" * 40)
        print("Starting constellation mapping...")
        input_bits = data.get(SignalGenerationStep.__output_key__)
        if input_bits is None:
            raise ValueError("Input bits not found in the data.")

        print(f"Input bits: {input_bits} (Total: {len(input_bits)} bits)")
        symbols = self.symbol_mapper.encode(input_bits)
        print(f"Total symbols: {len(symbols)}")
        print(f"Total constellation points: {len(self.symbol_mapper.constellation)}")
        print(f"Constellation energy: {np.mean(np.abs(self.symbol_mapper.constellation) ** 2):.4f}")
        print(f"Average symbol energy: {np.mean(np.abs(symbols) ** 2):.4f}")
        data[self.__output_key__] = symbols
        return super().execute(data)


class SerialParallelConversionStep(BaseSimulationStep):
    __output_key__ = "parallel_ofdm_symbols"

    def __init__(self, converter: SerialParallelConverter, num_subcarriers: int):
        self.converter = converter
        self.num_subcarriers = num_subcarriers

    def execute(self, data: ChainDataType) -> ChainDataType:
        """
        Convert the flat constellation symbols into OFDM symbols.
        """
        print("-" * 40)
        print("Starting serial to parallel conversion...")
        constellation_symbols = data.get(ConstellationMappingStep.__output_key__)
        if constellation_symbols is None:
            raise ValueError("Constellation symbols not found in the data.")

        print(f"Input shape: {constellation_symbols.shape}")
        ofdm_symbols = self.converter.to_parallel(constellation_symbols, self.num_subcarriers)
        print(
            f"Output shape: {ofdm_symbols.shape}, total: {ofdm_symbols.shape[0] * ofdm_symbols.shape[1]} symbols"
        )
        data[self.__output_key__] = ofdm_symbols
        return super().execute(data)


class ModulationStep(BaseSimulationStep):
    __output_key__ = "modulated_signal"

    def __init__(self, modulator: IModulator):
        self.modulator = modulator

    def execute(self, data: ChainDataType) -> ChainDataType:
        """
        Modulate the constellation symbols and add prefix.
        """
        print("-" * 40)
        print("Starting modulation...")
        parallel_constellation_symbols = data.get(SerialParallelConversionStep.__output_key__)
        if parallel_constellation_symbols is None:
            raise ValueError("Constellation symbols not found in the data.")

        print(f"Input shape: {parallel_constellation_symbols.shape}")
        modulated_signal = self.modulator.modulate(parallel_constellation_symbols)
        print(f"Output shape: {modulated_signal.shape}")
        print(f"Average modulated signal power: {np.mean(np.abs(modulated_signal) ** 2):.4f}")
        print(f"Peak modulated signal power: {np.max(np.abs(modulated_signal) ** 2):.4f}")
        print(
            f"PAPR (dB): {10 * np.log10(np.max(np.abs(modulated_signal) ** 2) / np.mean(np.abs(modulated_signal) ** 2)):.4f}"
        )
        data[self.__output_key__] = modulated_signal
        return super().execute(data)


class ChannelTransmissionStep(BaseSimulationStep):
    __output_key__ = "received_signal"

    def __init__(self, channel: ChannelModel, converter: SerialParallelConverter):
        self.channel = channel
        self.converter = converter

    def execute(self, data: ChainDataType) -> ChainDataType:
        """
        Transmit the modulated signal through the channel.
        """
        print("-" * 40)
        print("Starting channel transmission...")
        modulated_signal = data.get(ModulationStep.__output_key__)
        if modulated_signal is None:
            raise ValueError("Modulated signal not found in the data.")

        print(f"Input shape: {modulated_signal.shape}")

        serial_modulated_signal = self.converter.to_serial(modulated_signal)
        print(f"Serial modulated signal shape: {serial_modulated_signal.shape}")
        received_signal = self.channel.transmit(serial_modulated_signal)
        print(f"Received signal shape (serial): {received_signal.shape}")
        received_signal = self.converter.to_parallel(received_signal, modulated_signal.shape[1])
        print(f"Output shape: {received_signal.shape}")
        print(f"Average received signal power: {np.mean(np.abs(received_signal) ** 2):.4f}")
        print(f"Peak received signal power: {np.max(np.abs(received_signal) ** 2):.4f}")
        print(
            f"PAPR (dB): {10 * np.log10(np.max(np.abs(received_signal) ** 2) / np.mean(np.abs(received_signal) ** 2)):.4f}"
        )
        data[self.__output_key__] = received_signal
        return super().execute(data)


class DemodulationStep(BaseSimulationStep):
    __output_key__ = "demodulated_symbols"

    def __init__(self, modulator: IModulator):
        self.modulator = modulator

    def execute(self, data: ChainDataType) -> ChainDataType:
        """
        Demodulate the received signal and remove prefix.
        """
        print("-" * 40)
        print("Starting demodulation...")
        received_signal = data.get(ChannelTransmissionStep.__output_key__)
        if received_signal is None:
            raise ValueError("Received signal not found in the data.")

        print("Input shape: ", received_signal.shape)
        demodulated_symbols = self.modulator.demodulate(received_signal)
        print("Output shape: ", demodulated_symbols.shape)
        data[self.__output_key__] = demodulated_symbols
        return super().execute(data)


class CalculateNoiseVarianceStep(BaseSimulationStep):
    __output_key__ = "noise_variance"

    def __init__(self, channel: ChannelModel):
        self.snr_db = channel.snr_db
        self.channel = channel

    def execute(self, data: ChainDataType) -> ChainDataType:
        """
        Calculate the noise variance based on the SNR and channel characteristics.
        """
        print("-" * 40)
        print("Calculating noise variance...")
        # Convert SNR from dB to linear scale
        snr_linear = 10 ** (self.snr_db / 10)

        # Retrieve the transmitted signal power
        transmitted_signal = data.get(ChannelTransmissionStep.__output_key__)
        if transmitted_signal is None:
            raise ValueError("Transmitted signal not found in the data.")

        signal_power = np.mean(np.abs(transmitted_signal) ** 2)

        # Calculate noise power
        noise_power = signal_power / snr_linear

        # Adjust noise power based on channel gain if necessary
        channel_gain = np.mean(np.abs(self.channel.impulse_response) ** 2)
        if channel_gain > 0:
            noise_variance = noise_power / channel_gain
        else:
            noise_variance = noise_power

        print(f"Calculated noise variance: {noise_variance}")
        data[self.__output_key__] = noise_variance
        return super().execute(data)


class EqualizationStep(BaseSimulationStep):
    __output_key__ = "equalized_symbols"

    def __init__(self, equalizator: IEqualizator):
        self.equalizator = equalizator

    def execute(self, data: ChainDataType) -> ChainDataType:
        """
        Equalize the demodulated symbols.
        """
        print("-" * 40)
        print("Starting equalization...")
        demodulated_symbols = data.get(DemodulationStep.__output_key__)
        if demodulated_symbols is None:
            raise ValueError("Demodulated symbols not found in the data.")

        noise_variance = data.get(CalculateNoiseVarianceStep.__output_key__, None)
        print("Noise variance: ", noise_variance)

        equalized_symbols = self.equalizator.equalize(demodulated_symbols, noise_variance)
        print("Equalized symbols shape: ", equalized_symbols.shape)
        print("Average equalized symbol power: ", np.mean(np.abs(equalized_symbols) ** 2))
        print("Peak equalized symbol power: ", np.max(np.abs(equalized_symbols) ** 2))
        print(
            "PAPR (dB): ",
            10
            * np.log10(
                np.max(np.abs(equalized_symbols) ** 2) / np.mean(np.abs(equalized_symbols) ** 2)
            ),
        )
        data[self.__output_key__] = equalized_symbols
        return super().execute(data)


class ParallelSerialConversionStep(BaseSimulationStep):
    __output_key__ = "serial_symbols"

    def __init__(self, converter: SerialParallelConverter):
        self.converter = converter

    def execute(self, data: ChainDataType) -> ChainDataType:
        """
        Convert the parallel OFDM symbols back to a serial stream.
        """
        print("-" * 40)
        print("Starting parallel to serial conversion...")
        parallel_ofdm_symbols = data.get(EqualizationStep.__output_key__)
        if parallel_ofdm_symbols is None:
            raise ValueError("Equalized symbols not found in the data.")

        print("Input shape: ", parallel_ofdm_symbols.shape)
        serial_symbols = self.converter.to_serial(parallel_ofdm_symbols)
        print("Output shape: ", serial_symbols.shape)
        data[self.__output_key__] = serial_symbols
        return super().execute(data)


class ConstellationDemappingStep(BaseSimulationStep):
    __output_key__ = "output_bits"

    def __init__(self, symbol_mapper: IConstellationSymbolMapper):
        self.symbol_mapper = symbol_mapper

    def execute(self, data: ChainDataType) -> ChainDataType:
        """
        Map the received constellation symbols back to bits.
        """
        print("-" * 40)
        print("Starting constellation demapping...")
        serial_symbols = data.get(ParallelSerialConversionStep.__output_key__)
        if serial_symbols is None:
            raise ValueError("Serial symbols not found in the data.")

        print(f"Input symbols length: {len(serial_symbols)}")
        print(f"Constellation points length: {len(self.symbol_mapper.constellation)}")
        output_bits = self.symbol_mapper.decode(serial_symbols)
        print(f"Output bits: {output_bits} (Total: {len(output_bits)} bits)")
        data[self.__output_key__] = output_bits
        return super().execute(data)


class CalculateBERStep(BaseSimulationStep):
    __output_key__ = "ber"

    def __init__(self):
        pass

    def execute(self, data: ChainDataType) -> ChainDataType:
        """
        Calculate the Bit Error Rate (BER) between input and output bits.
        """
        print("-" * 40)
        print("Calculating BER...")
        input_bits = data.get(SignalGenerationStep.__output_key__)
        output_bits = data.get(ConstellationDemappingStep.__output_key__)

        if input_bits is None:
            raise ValueError("Input bits not found in the data.")
        if output_bits is None:
            raise ValueError("Output bits not found in the data.")
        if len(input_bits) != len(output_bits):
            raise ValueError(
                f"Input and output bits must have the same length: o:{len(output_bits)}, i:{len(input_bits)}"
            )

        # Calculate BER
        num_errors = np.sum(np.array(input_bits) != np.array(output_bits))
        ber = num_errors / len(input_bits)

        print(f"Calculated BER: {ber} ({num_errors} errors out of {len(input_bits)} bits)")
        data[self.__output_key__] = ber
        return super().execute(data)


class CalculateSERStep(BaseSimulationStep):
    __output_key__ = "ser"

    def __init__(self, symbol_mapper: IConstellationSymbolMapper):
        self.symbol_mapper = symbol_mapper

    def execute(self, data: ChainDataType) -> ChainDataType:
        """
        Calculate the Symbol Error Rate (SER) between input and output symbols.
        """
        print("-" * 40)
        print("Calculating SER...")
        input_symbols = data.get(ConstellationMappingStep.__output_key__)
        output_symbols = data.get(ParallelSerialConversionStep.__output_key__)

        if input_symbols is None:
            raise ValueError("Input symbols not found in the data.")
        if output_symbols is None:
            raise ValueError("Output symbols not found in the data.")
        if len(input_symbols) != len(output_symbols):
            raise ValueError("Input and output symbols must have the same length.")

        # Map output symbols back to the nearest constellation points
        # This is necessary because the output symbols may not exactly match the constellation point
        # So, the trick is to decode and then encode again, retrieving the clean constellation point
        # For example:
        # 1.30948 -> b1
        # b1 -> 1.3
        decided_output_symbols = self.symbol_mapper.decode(output_symbols)
        decided_output_symbols = self.symbol_mapper.encode(decided_output_symbols)

        # Calculate SER
        num_errors = np.sum(np.array(input_symbols) != np.array(decided_output_symbols))
        ser = num_errors / len(input_symbols)

        print(f"Calculated SER: {ser} ({num_errors} errors out of {len(input_symbols)} symbols)")
        data[self.__output_key__] = ser
        return super().execute(data)


class CalculateSNRStep(BaseSimulationStep):
    __output_key__ = "snr"

    def __init__(self, channel: ChannelModel):
        self.channel = channel

    def execute(self, data: ChainDataType) -> ChainDataType:
        """
        Calculate the Signal-to-Noise Ratio (SNR) of the received signal.
        """
        print("-" * 40)
        print("Calculating SNR...")
        modulated_signal = data.get(ModulationStep.__output_key__)
        received_signal = data.get(ChannelTransmissionStep.__output_key__)

        if modulated_signal is None:
            raise ValueError("Modulated signal not found in the data.")
        if received_signal is None:
            raise ValueError("Received signal not found in the data.")

        # Calculate signal power
        signal_power = np.mean(np.abs(modulated_signal) ** 2)

        # Noise power is the noise variance calculated earlier
        noise_variance = data.get(CalculateNoiseVarianceStep.__output_key__, None)
        if noise_variance is None:
            raise ValueError("Noise variance not found in the data.")

        noise_power = noise_variance

        if noise_power == 0:
            raise ValueError("Noise power is zero, cannot calculate SNR.")

        # Calculate SNR in dB
        snr = 10 * np.log10(signal_power / noise_power)

        print(
            f"Calculated SNR: {snr} dB (Signal Power: {signal_power}, Noise Power: {noise_power})"
        )
        data[self.__output_key__] = snr
        return super().execute(data)


class CalculatePAPRStep(BaseSimulationStep):
    __output_key__ = "papr"

    def __init__(self):
        pass

    def execute(self, data: ChainDataType) -> ChainDataType:
        """
        Calculate the Peak-to-Average Power Ratio (PAPR) of the modulated signal.
        """
        print("-" * 40)
        print("Calculating PAPR...")
        modulated_signal = data.get(ModulationStep.__output_key__)

        if modulated_signal is None:
            raise ValueError("Modulated signal not found in the data.")

        # Calculate peak power
        peak_power = np.max(np.abs(modulated_signal) ** 2)

        # Calculate average power
        average_power = np.mean(np.abs(modulated_signal) ** 2)

        if average_power == 0:
            raise ValueError("Average power is zero, cannot calculate PAPR.")

        # Calculate PAPR in dB
        papr = 10 * np.log10(peak_power / average_power)

        print(
            f"Calculated PAPR: {papr} dB (Peak Power: {peak_power}, Average Power: {average_power})"
        )
        data[self.__output_key__] = papr
        return super().execute(data)


class PlotConstellationStep(BaseSimulationStep):
    __output_key__ = "constellation_plot"

    def __init__(
        self,
        symbol_mapper: IConstellationSymbolMapper,
    ):
        self.symbol_mapper = symbol_mapper

    def execute(self, data: ChainDataType) -> ChainDataType:
        """
        Plot the constellation diagram of the received symbols.
        """
        print("-" * 40)
        print("Plotting constellation diagram...")
        serial_symbols = data.get(ParallelSerialConversionStep.__output_key__)
        ber = data.get(CalculateBERStep.__output_key__, None)
        ser = data.get(CalculateSERStep.__output_key__, None)
        snr = data.get(CalculateSNRStep.__output_key__, None)
        papr = data.get(CalculatePAPRStep.__output_key__, None)
        if serial_symbols is None:
            raise ValueError("Serial symbols not found in the data.")

        # Clear previous plots
        plt.clf()
        plt.cla()

        # Plot received constellation
        plt.figure(figsize=(8, 8))
        plt.scatter(
            serial_symbols.real,
            serial_symbols.imag,
            color="blue",
            alpha=0.5,
            label="Received Symbols",
        )

        # Plot ideal constellation points
        ideal_points = self.symbol_mapper.constellation
        plt.scatter(
            ideal_points.real,
            ideal_points.imag,
            color="red",
            marker="x",
            s=100,
            label="Ideal Constellation Points",
        )

        # Set plot attributes
        plt.title("Constellation Diagram")
        plt.xlabel("In-Phase")
        plt.ylabel("Quadrature")
        plt.axhline(0, color="black", linewidth=0.5)
        plt.axvline(0, color="black", linewidth=0.5)
        plt.legend(loc="upper right")  # Specify a fixed location for the legend
        plt.grid(True)
        plt.xlim(-1.5, 1.5)
        plt.ylim(-1.5, 1.5)
        plt.grid(False)  # Remove the inner grid
        plt.axis("equal")

        # Add text box with BER, SER, SNR, and PAPR
        textstr = ""
        if ber is not None:
            textstr += f"BER: {ber:.4e}\n"
        if ser is not None:
            textstr += f"SER: {ser:.4e}\n"
        if snr is not None:
            textstr += f"SNR: {snr:.2f} dB\n"
        if papr is not None:
            textstr += f"PAPR: {papr:.2f} dB\n"
        props = {"boxstyle": "round", "facecolor": "white", "alpha": 0.8}
        plt.gcf().text(0.02, 0.98, textstr, fontsize=10, verticalalignment="top", bbox=props)
        plt.tight_layout()

        # Save the plot to a BytesIO object instead of the filesystem
        buffer = BytesIO()
        plt.savefig(buffer, format="png", dpi=300)
        buffer.seek(0)

        # Open the image directly from the buffer with Pillow
        img = Image.open(buffer)

        # Return the image object in the data dictionary
        data[self.__output_key__] = img

        # Close the plot to free memory
        plt.close()
        print("Constellation diagram plotted and image saved.")
        return super().execute(data)


class SimulationChain:
    def __init__(self, first_step: ISimulationStep):
        self.first_step = first_step

    def run(self, initial_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        if initial_data is None:
            initial_data = {}
        return self.first_step.execute(initial_data)


class Simulation:
    def __init__(
        self,
        configuration: SimulationSettings,
        bits_generator: BitsGenerator,
        symbol_mapper: IConstellationSymbolMapper,
        serial_parallel_converter: SerialParallelConverter,
        channel: ChannelModel,
        modulator: IModulator,
        equalizator: IEqualizator,
    ):
        self.configuration = configuration
        self.bits_generator = bits_generator
        self.symbol_mapper = symbol_mapper
        self.serial_parallel_converter = serial_parallel_converter
        self.modulator = modulator
        self.channel = channel
        self.equalizator = equalizator
        self.metadata: Dict[str, Any] = {}
        self.chain: Optional[SimulationChain] = None
        self.build_metadata()
        self.setup_chain()

    def setup_chain(self) -> None:
        # ------------------------------------------------------------------------------------------
        # Signal Generation Step
        # ------------------------------------------------------------------------------------------
        signal_generations_step = SignalGenerationStep(
            bits_generator=self.bits_generator,
            num_bits=self.configuration.num_bits,
            num_symbols=self.configuration.num_symbols,
            constellation_order=self.configuration.constellation_order,
        )

        # ------------------------------------------------------------------------------------------
        # Constellation Mapping Step
        # ------------------------------------------------------------------------------------------
        constellation_mapping_step = ConstellationMappingStep(symbol_mapper=self.symbol_mapper)
        signal_generations_step.set_next(constellation_mapping_step)

        # ------------------------------------------------------------------------------------------
        # Serial to Parallel Conversion Step
        # ------------------------------------------------------------------------------------------
        serial_to_parallel_step = SerialParallelConversionStep(
            converter=self.serial_parallel_converter,
            num_subcarriers=self.configuration.num_bands,
        )
        constellation_mapping_step.set_next(serial_to_parallel_step)

        # ------------------------------------------------------------------------------------------
        # Modulation Step
        # ------------------------------------------------------------------------------------------
        modulation_step = ModulationStep(modulator=self.modulator)
        serial_to_parallel_step.set_next(modulation_step)

        # ------------------------------------------------------------------------------------------
        # Channel Transmission Step
        # ------------------------------------------------------------------------------------------
        channel_transmission_step = ChannelTransmissionStep(
            channel=self.channel,
            converter=self.serial_parallel_converter,
        )
        modulation_step.set_next(channel_transmission_step)

        # ------------------------------------------------------------------------------------------
        # Demodulation Step
        # ------------------------------------------------------------------------------------------
        demodulation_step = DemodulationStep(modulator=self.modulator)
        channel_transmission_step.set_next(demodulation_step)

        # ------------------------------------------------------------------------------------------
        # Noise Variance Calculation Step
        # ------------------------------------------------------------------------------------------
        noise_variance_step = CalculateNoiseVarianceStep(
            channel=self.channel,
        )
        demodulation_step.set_next(noise_variance_step)

        # ------------------------------------------------------------------------------------------
        # Equalization Step
        # ------------------------------------------------------------------------------------------
        equalization_step = EqualizationStep(equalizator=self.equalizator)
        noise_variance_step.set_next(equalization_step)

        # ------------------------------------------------------------------------------------------
        # Parallel to Serial Conversion Step
        # ------------------------------------------------------------------------------------------
        parallel_to_serial_step = ParallelSerialConversionStep(
            converter=self.serial_parallel_converter,
        )
        equalization_step.set_next(parallel_to_serial_step)

        # ------------------------------------------------------------------------------------------
        # Constellation Demapping Step
        # ------------------------------------------------------------------------------------------
        constellation_demapping_step = ConstellationDemappingStep(
            symbol_mapper=self.symbol_mapper,
        )
        parallel_to_serial_step.set_next(constellation_demapping_step)

        # ------------------------------------------------------------------------------------------
        # Calculate BER Step
        # ------------------------------------------------------------------------------------------
        calculate_ber_step = CalculateBERStep()
        constellation_demapping_step.set_next(calculate_ber_step)

        # ------------------------------------------------------------------------------------------
        # Calculate SER Step
        # ------------------------------------------------------------------------------------------
        calculate_ser_step = CalculateSERStep(symbol_mapper=self.symbol_mapper)
        calculate_ber_step.set_next(calculate_ser_step)

        # ------------------------------------------------------------------------------------------
        # Calculate SNR Step
        # ------------------------------------------------------------------------------------------
        calculate_snr_step = CalculateSNRStep(channel=self.channel)
        calculate_ser_step.set_next(calculate_snr_step)

        # ------------------------------------------------------------------------------------------
        # Calculate PAPR Step
        # ------------------------------------------------------------------------------------------
        calculate_papr_step = CalculatePAPRStep()
        calculate_snr_step.set_next(calculate_papr_step)

        # ------------------------------------------------------------------------------------------
        # Plot Constellation Step
        # ------------------------------------------------------------------------------------------
        plot_constellation_step = PlotConstellationStep(
            symbol_mapper=self.symbol_mapper,
        )
        calculate_snr_step.set_next(plot_constellation_step)

        # ------------------------------------------------------------------------------------------
        # Finalize the chain
        # ------------------------------------------------------------------------------------------
        self.chain = SimulationChain(first_step=signal_generations_step)

    def build_metadata(self) -> None:
        file_path = f"images/{self.channel.channel_type.lower()}_channel/"
        filename = (
            f"{self.configuration.constellation_type}_"
            f"{self.configuration.constellation_order}_"
            f"{self.configuration.prefix_type}_"
            f"{self.configuration.equalization_method}_"
            f"{self.configuration.noise_type}_"
            f"{self.channel.snr_db}dB.png"
        )
        self.metadata = {
            "configuration": self.configuration.model_dump(),
            "channel_impulse_response": self.channel.impulse_response.tolist(),
            "channel_type": self.channel.channel_type,
            "snr_db": self.channel.snr_db,
            "noise_model": type(self.channel.noise_model).__name__,
            "constellation": {
                "type": self.configuration.constellation_type,
                "order": self.configuration.constellation_order,
                "points": self.symbol_mapper.constellation.tolist(),
            },
            "modulation": {
                "type": type(self.modulator).__name__,
                "fft_size": self.configuration.num_bands,
                "prefix_type": self.configuration.prefix_type,
            },
            "equalizator": type(self.equalizator).__name__,
            "bits_generator": type(self.bits_generator).__name__,
            "images": {
                "constellation_diagram_filename": filename,
                "constellation_diagram_path": file_path,
                "constellation_diagram_full_path": f"{file_path}{filename}",
            },
        }

    def run(self) -> Dict[str, Any]:
        if self.chain is None:
            self.setup_chain()

        if self.chain is None:
            raise RuntimeError("Simulation chain is not properly set up.")

        result = self.chain.run(initial_data={})
        result["metadata"] = self.metadata

        return result
