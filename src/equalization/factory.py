from equalization.base import OFDMZeroForcingEqualizator, OFDMMMSEEqualizator, NoEqualizator
from configuration.enums import EqualizationMethod


class EqualizationFactory:
    EQUALIZATION_METHODS = {
        EqualizationMethod.ZF: OFDMZeroForcingEqualizator,
        EqualizationMethod.MMSE: OFDMMMSEEqualizator,
        EqualizationMethod.NONE: NoEqualizator,
    }

    @classmethod
    def create_equalizator(
        cls, method: EqualizationMethod, channel=None, fft_size=None, noise_variance=None
    ):
        equalizator_class = cls.EQUALIZATION_METHODS.get(method)
        if not equalizator_class:
            raise ValueError(f"Unsupported equalization method: {method}")

        if method == EqualizationMethod.MMSE and noise_variance is None:
            raise ValueError("noise_variance must be provided for MMSE equalization.")

        if method in (EqualizationMethod.ZF, EqualizationMethod.MMSE):
            if channel is None or fft_size is None:
                raise ValueError(
                    f"channel and fft_size must be provided for {method} equalization."
                )

        return equalizator_class(channel=channel, fft_size=fft_size, noise_variance=noise_variance)
