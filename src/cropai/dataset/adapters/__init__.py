"""Source adapters convert external datasets into ImageRecord lists."""

from .base import SourceAdapter, discover_imagefolder, load_class_map
from .plantvillage import PlantVillageAdapter
from .plantdoc import PlantDocAdapter
from .ip102 import IP102Adapter
from .field_csv import FieldCSVAdapter

ADAPTERS = {
    "plantvillage": PlantVillageAdapter,
    "plantdoc": PlantDocAdapter,
    "ip102": IP102Adapter,
    "field_csv": FieldCSVAdapter,
    "cropsap_csv": FieldCSVAdapter,
}


def get_adapter(name: str) -> SourceAdapter:
    if name not in ADAPTERS:
        raise KeyError(f"unknown adapter {name}. known={sorted(ADAPTERS)}")
    return ADAPTERS[name]()
