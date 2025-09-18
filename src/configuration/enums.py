from enum import Enum


class ConstellationType(str, Enum):
    QAM = "QAM"
    PSK = "PSK"

    def __str__(self):
        return self.value


class PrefixType(str, Enum):
    CYCLIC = "CYCLIC"
    ZERO = "ZERO"
    NONE = "NONE"

    def __str__(self):
        return self.value


class EqualizationMethod(str, Enum):
    ZF = "ZF"
    MMSE = "MMSE"
    NONE = "NONE"

    def __str__(self):
        return self.value
