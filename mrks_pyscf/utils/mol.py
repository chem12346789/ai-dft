"""Molecular dict"""
H_2 = [["H", 100, 0, 0], ["H", 0, 0, 0]]
Be = [["Be", 0, 0, 0]]
HF = [["H", 100, 0, 0], ["F", 0, 0, 0]]
HCN = [
    ["H", 100, 0, 0],
    ["C", 0, 0, 0],
    ["N", -1.1560, 0, 0],
]
HOH = [
    ["H", 1.0, 0.0, 0.0],
    ["O", 0.0, 0.0, 0.0],
    ["H", -0.2438, 0.9698, 0.0],
]

Mol = {
    "H_2": H_2,
    "HH": H_2,
    "HF": HF,
    "Be": Be,
    "HCN": HCN,
    "HOH": HOH,
}

BASIS = {
    "augccpwcv6z": "aug-cc-pv6z",
    "augccpwcv5z": "aug-cc-pv5z",
    "augccpwcvqz": "aug-cc-pvqz",
    "augccpwcvtz": "aug-cc-pvtz",
    "augccpwcvdz": "aug-cc-pvdz",
    "cc-pcv6z": "cc-pv6z",
    "cc-pcv5z": "cc-pv5z",
    "cc-pcvqz": "cc-pvqz",
    "cc-pcvtz": "cc-pvtz",
    "cc-pcvdz": "cc-pvdz",
}


def old_function1(distance):
    """
    This function is used to determine the factor of mixing old and new density matrix in SCF process
    """
    if distance < 1.5:
        return 0.8
    if distance < 2.5:
        return 0.9
    if distance < 3.5:
        return 0.95


def old_function2(distance):
    """
    This function is used to determine the factor of mixing old and new density matrix in SCF process
    """
    if distance < 1.5:
        return 0.9
    if distance < 2.5:
        return 0.95
    if distance < 3.5:
        return 0.975

