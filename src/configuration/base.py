import os
import json
from typing import Optional

from pydantic import BaseModel, Field, field_validator
from configuration.enums import ConstellationType, PrefixType
from channel.enums import NoiseType, ChannelType


class BaseSettings(BaseModel):
    """Base settings class to load configuration from a JSON file."""

    @classmethod
    def from_json(cls, file_path: str):
        """Load settings from a JSON file."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Configuration file not found: {file_path}")
        with open(file_path, "r", encoding="utf-8") as file:
            data = json.load(file)
        return cls(**data)


class Settings(BaseSettings):
    """Base configuration class for the project."""

    project_name: str = Field(..., description="The name of the project")
    version: str = Field(..., description="The version of the project")
    debug: bool = Field(False, description="Enable or disable debug mode")

    def __str__(self):
        return f"{self.project_name}\n{self.version}\nDebug Mode: {self.debug}"


class SimulationSettings(BaseSettings):
    """Configuration class for simulation parameters."""

    num_bands: int = Field(..., description="Number of frequency bands")
    signal_noise_ratios: list[float] = Field(
        ..., description="List of signal-to-noise ratios for simulation"
    )
    channel_model_path: str = Field(..., description="Path to the channel model file")
    channel_type: ChannelType = Field(
        ChannelType.FLAT, description="Type of the channel (e.g., FLAT or CUSTOM)"
    )
    noise_type: NoiseType = Field(
        NoiseType.AWGN, description="Type of noise to be added (e.g., AWGN, NONE)"
    )
    num_bits: Optional[int] = Field(None, description="Number of bits to simulate")
    num_symbols: Optional[int] = Field(
        None, description="Number of symbols to simulate"
    )
    constellation_order: int = Field(
        16, description="Order of the QAM constellation (e.g., 4, 16, 64)"
    )
    constellation_type: ConstellationType = Field(
        ConstellationType.PSK,
        description="Type of the constellation (e.g., QAM or PSK)",
    )
    prefix_type: PrefixType = Field(
        PrefixType.CYCLIC, description="Type of cyclic prefix (e.g., CYCLIC or ZERO)"
    )
    prefix_length_ratio: float = Field(
        0.25,
        description="Ratio of cyclic prefix length to channel time domain size",
    )

    def __str__(self):
        return (
            f"Number of Bands: {self.num_bands}\n"
            f"Signal-to-Noise Ratios: {self.signal_noise_ratios}\n"
            f"Channel Type: {self.channel_type}\n"
            f"Channel Model Path: '{self.channel_model_path}'\n"
            f"Noise Type: {self.noise_type}\n"
            f"{'Number of Bits: ' + str(self.num_bits) if self.num_bits is not None else ''}\n"
            f"{'Number of Symbols: ' + str(self.num_symbols) if self.num_symbols is not None else ''}\n"
            f"Constellation Type: '{self.constellation_type}'\n"
            f"Constellation Order: {self.constellation_order}\n"
            f"Prefix Type: {self.prefix_type}\n"
            f"Prefix Length Ratio: {self.prefix_length_ratio}"
        )

    @field_validator("num_symbols")
    @classmethod
    def check_bits_or_symbols(cls, v, info):
        if info.data.get("num_bits") is None and v is None:
            raise ValueError("Either num_bits or num_symbols must be specified.")
        if info.data.get("num_bits") is not None and v is not None:
            raise ValueError("Only one of num_bits or num_symbols should be specified.")
        return v

    @field_validator("prefix_length_ratio")
    @classmethod
    def validate_prefix_length_ratio(cls, v):
        if not (0.0 <= v <= 1.0):
            raise ValueError("prefix_length_ratio must be between 0 and 1 (inclusive).")
        return v
