"""Molecular dict"""

import importlib.resources
from pathlib import Path
import os
import json

H_2 = [["H", 0.737, 0, 0], ["H", 0, 0, 0]]
Be = [["Be", 0, 0, 0]]
Ne = [["Ne", 0, 0, 0]]
He = [["He", 0, 0, 0]]
HNHH = [
    ["H", 1.008, -0.0, 0.0],
    ["N", 0.0, 0.0, 0.0],
    ["H", -0.2906, 0.9122, 0.3151],
    ["H", -0.2903, -0.0808, -0.962],
]
HOH = [
    ["H", 1.0, 0.0, 0.0],
    ["O", 0.0, 0.0, 0.0],
    ["H", -0.2438, 0.9698, 0.0],
]
HOOH = [
    ["H", 0.9688, -0.0, 0.0],
    ["O", 0.0, 0.0, 0.0],
    ["O", -0.241, 0.4457, 1.3617],
    ["H", -0.6906, 1.2843, 1.179],
]

Allene = [
    ["C", 3.998, 0.660, -0.000],
    ["C", 1.734, -0.660, -0.000],
    ["C", 2.866, 0.000, -0.001],
    ["H", 4.822, 0.375, -0.661],
    ["H", 4.153, 1.518, 0.662],
    ["H", 0.910, -0.375, -0.661],
    ["H", 1.579, -1.518, 0.662],
]
Cyclopropene = [
    ["C", -0.890, -0.000, 0.000],
    ["C", 0.477, 0.650, -0.000],
    ["C", 0.477, -0.650, -0.000],
    ["H", -1.494, -0.000, 0.921],
    ["H", -1.494, -0.000, -0.921],
    ["H", 1.023, 1.588, 0.000],
    ["H", 1.024, -1.588, 0.000],
]
Propyne = [
    ["C", -1.375, 0.000, -0.000],
    ["C", 0.085, 0.000, 0.000],
    ["C", 1.297, 0.000, -0.000],
    ["H", -1.772, 0.066, 1.026],
    ["H", -1.772, 0.856, -0.570],
    ["H", -1.772, -0.922, -0.456],
    ["H", 2.369, 0.000, -0.000],
]
Cyclopropane = [
    ["C", 0.321, 0.811, 0.000],
    ["C", -0.863, -0.128, -0.000],
    ["C", 0.543, -0.684, -0.000],
    ["H", 0.539, 1.363, -0.918],
    ["H", 0.539, 1.363, 0.918],
    ["H", -1.450, -0.215, 0.918],
    ["H", -1.449, -0.215, -0.918],
    ["H", 0.911, -1.148, -0.918],
    ["H", 0.911, -1.148, 0.918],
]
Propylene = [
    ["C", 1.289, -0.200, -0.000],
    ["C", -0.073, 0.430, -0.000],
    ["C", -1.233, -0.233, 0.000],
    ["H", 1.872, 0.113, -0.884],
    ["H", 1.229, -1.300, -0.000],
    ["H", 1.872, 0.113, 0.884],
    ["H", -0.094, 1.528, -0.000],
    ["H", -2.192, 0.291, 0.000],
    ["H", -1.265, -1.328, 0.000],
]
Propane = [
    ["C", 0.000, -0.565, -0.000],
    ["C", -1.277, 0.279, -0.000],
    ["C", 1.277, 0.279, 0.000],
    ["H", 0.000, -1.231, 0.882],
    ["H", 0.000, -1.231, -0.882],
    ["H", -1.324, 0.932, 0.889],
    ["H", -1.324, 0.932, -0.889],
    ["H", -2.181, -0.351, -0.000],
    ["H", 2.181, -0.351, 0.000],
    ["H", 1.324, 0.932, 0.889],
    ["H", 1.324, 0.932, -0.889],
]
Bicyclobutane = [
    ["C", 0.000, 0.749, -0.344],
    ["C", -0.000, -0.749, -0.344],
    ["C", 1.139, 0.000, 0.287],
    ["C", -1.139, 0.000, 0.287],
    ["H", 0.000, 1.436, -1.187],
    ["H", -0.000, -1.436, -1.187],
    ["H", 2.091, -0.000, -0.256],
    ["H", 1.244, 0.000, 1.382],
    ["H", -1.244, 0.000, 1.382],
    ["H", -2.091, 0.000, -0.256],
]

Mol = {
    "H_2": H_2,
    "HH": H_2,
    "Be": Be,
    "He": He,
    "Ne": Ne,
    "HOH": HOH,
    "HOOH": HOOH,
    "HNHH": HNHH,
    "HNH2": HNHH,
    "allene": Allene,
    "cyclopropene": Cyclopropene,
    "propyne": Propyne,
    "cyclopropane": Cyclopropane,
    "propylene": Propylene,
    "propane": Propane,
    # "isobutane": Isobutane,
    # "butane": Butane,
    # "butadiene": Butadiene,
    # "butyne": Butyne,
    "bicyclobutane": Bicyclobutane,
    # "cyclopropylmethyl": Cyclopropylmethyl,
    # "cyclobutane": Cyclobutane,
    # "spiropentane": Spiropentane,
    # "benzene": Benzene,
    # "pentane": Pentane,
    # "isopentane": Isopentane,
    # "neopentane": Neopentane,
}

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

# HASH_LIST = [0, 1]
# ATOM_HASH_DICT = {"H": 0, "C": 1}

HASH_LIST = [0]
ATOM_HASH_DICT = {"H": 0, "C": 0}
