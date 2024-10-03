"""Molecular dict"""

import importlib.resources
from pathlib import Path
import os
import json

Mol = {}

with importlib.resources.path("cadft", "utils") as resource_path:
    with open(
        Path(os.fspath(resource_path)) / "mol.json",
        "r",
        encoding="utf-8",
    ) as f:
        Mol.update(json.load(f))

name_mol = []

for name in Mol:
    name_mol.append(name)
