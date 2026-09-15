from pathlib import Path
from typing import Union

import yaml


def load_config(path: Union[str, Path]) -> dict:
    with open(path) as f:
        return yaml.safe_load(f) or {}
