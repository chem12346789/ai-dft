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
Isobutane = [
    ["C", -0.002, -0.001, -0.353],
    ["C", -0.540, 1.358, 0.114],
    ["C", 1.443, -0.214, 0.115],
    ["C", -0.909, -1.146, 0.115],
    ["H", -0.001, -0.001, -1.461],
    ["H", -0.563, 1.414, 1.217],
    ["H", 0.090, 2.187, -0.249],
    ["H", -1.567, 1.531, -0.249],
    ["H", 1.502, -0.222, 1.218],
    ["H", 2.107, 0.589, -0.247],
    ["H", 1.848, -1.174, -0.247],
    ["H", -0.946, -1.193, 1.218],
    ["H", -0.546, -2.122, -0.247],
    ["H", -1.942, -1.016, -0.248],
]
Butane = [
    ["C", -0.575, 0.506, -0.001],
    ["C", 0.575, -0.506, -0.001],
    ["C", -1.959, -0.148, -0.001],
    ["C", 1.959, 0.148, -0.001],
    ["H", -0.478, 1.166, -0.883],
    ["H", -0.478, 1.166, 0.881],
    ["H", 0.478, -1.166, 0.881],
    ["H", 0.478, -1.166, -0.884],
    ["H", -2.101, -0.785, -0.891],
    ["H", -2.100, -0.787, 0.887],
    ["H", -2.764, 0.605, -0.000],
    ["H", 2.101, 0.785, -0.891],
    ["H", 2.764, -0.605, -0.000],
    ["H", 2.100, 0.787, 0.887],
]
Butadiene = [
    ["C", -0.613, 0.396, 0.000],
    ["C", 0.612, -0.395, 0.000],
    ["C", -1.850, -0.125, 0.000],
    ["C", 1.851, 0.125, 0.000],
    ["H", -0.487, 1.486, 0.000],
    ["H", 0.487, -1.485, 0.000],
    ["H", -2.738, 0.510, 0.000],
    ["H", -2.012, -1.208, 0.000],
    ["H", 2.738, -0.511, 0.000],
    ["H", 2.012, 1.207, 0.000],
]
Butyne = [
    ["C", -2.069, -0.000, 0.000],
    ["C", 2.069, -0.000, 0.000],
    ["C", -0.607, -0.000, -0.000],
    ["C", 0.607, -0.000, -0.000],
    ["H", -2.470, -0.339, 0.970],
    ["H", -2.470, 1.009, -0.192],
    ["H", -2.470, -0.671, -0.778],
    ["H", 2.470, -1.027, 0.024],
    ["H", 2.470, 0.534, 0.877],
    ["H", 2.470, 0.492, -0.901],
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
Cyclopropylmethyl = [
    ["C", 0.933, 0.772, -0.000],
    ["C", 0.933, -0.772, 0.000],
    ["C", -0.320, -0.000, -0.000],
    ["C", -1.645, 0.000, -0.000],
    ["H", 1.238, 1.282, -0.920],
    ["H", 1.237, 1.282, 0.920],
    ["H", 1.237, -1.282, 0.920],
    ["H", 1.237, -1.282, -0.920],
    ["H", -2.212, -0.935, 0.000],
    ["H", -2.212, 0.935, -0.000],
]
Cyclobutane = [
    ["C", 0.127, -1.078, 0.122],
    ["C", -1.078, -0.127, -0.122],
    ["C", 1.078, 0.127, -0.122],
    ["C", -0.127, 1.078, 0.122],
    ["H", 0.168, -1.422, 1.168],
    ["H", 0.231, -1.954, -0.536],
    ["H", -1.422, -0.168, -1.168],
    ["H", -1.954, -0.231, 0.537],
    ["H", 1.955, 0.231, 0.536],
    ["H", 1.422, 0.168, -1.168],
    ["H", -0.231, 1.955, -0.536],
    ["H", -0.168, 1.422, 1.168],
]
Spiropentane = [
    ["C", 0.000, 0.000, 0.000],
    ["C", -1.274, -0.518, -0.564],
    ["C", -1.274, 0.518, 0.564],
    ["C", 1.274, 0.565, -0.518],
    ["C", 1.274, -0.565, 0.518],
    ["H", -1.577, -1.539, -0.314],
    ["H", -1.578, -0.182, -1.560],
    ["H", -1.578, 1.539, 0.315],
    ["H", -1.578, 0.183, 1.560],
    ["H", 1.578, 0.315, -1.539],
    ["H", 1.578, 1.560, -0.182],
    ["H", 1.578, -1.560, 0.182],
    ["H", 1.578, -0.315, 1.539],
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
    "isobutane": Isobutane,
    "butane": Butane,
    "butadiene": Butadiene,
    "butyne": Butyne,
    "bicyclobutane": Bicyclobutane,
    "cyclopropylmethyl": Cyclopropylmethyl,
    "cyclobutane": Cyclobutane,
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
