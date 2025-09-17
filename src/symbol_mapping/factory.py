from configuration.enums import ConstellationType
from symbol_mapping.base import (
    IConstellationSymbolMapper,
    PSKConstellationSymbolMapper,
    QAMConstellationSymbolMapper,
)

CONSTELLATION_MAPPERS = {
    ConstellationType.QAM: QAMConstellationSymbolMapper,
    ConstellationType.PSK: PSKConstellationSymbolMapper,
}


class SymbolMapperFactory:
    @staticmethod
    def create_mapper(
        constellation_type: ConstellationType, order: int
    ) -> IConstellationSymbolMapper:
        mapper_class = CONSTELLATION_MAPPERS.get(constellation_type)
        if not mapper_class:
            raise ValueError(f"Unsupported constellation type: {constellation_type}")
        return mapper_class(order)
