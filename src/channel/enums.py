from enum import Enum


class ChannelType(str, Enum):
    FLAT = "FLAT"
    CUSTOM = "CUSTOM"

    def __str__(self):
        return self.value


class NoiseType(str, Enum):
    AWGN = "AWGN"
    NONE = "NONE"

    def __str__(self):
        return self.value
