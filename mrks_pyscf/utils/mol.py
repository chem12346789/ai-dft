"""Molecular dict"""
H_2 = [["H", 1, 0, 0], ["H", -1, 0, 0]]
Be = [["Be", 0, 0, 0]]
HF = [["H", 1, 0, 0], ["F", -1, 0, 0]]

Mol = {"H_2": H_2, "HH": H_2, "HF": HF, "Be": Be}

BASIS = {
    "augccpwcv6z": "aug-cc-pv6z",
    "augccpwcv5z": "aug-cc-pv5z",
    "augccpwcvqz": "aug-cc-pvqz",
    "augccpwcvtz": "aug-cc-pvtz",
    "augccpwcvdz": "aug-cc-pvdz",
    'cc-pcv6z': 'cc-pv6z',
    'cc-pcv5z': 'cc-pv5z',
    'cc-pcvqz': 'cc-pvqz',
    'cc-pcvtz': 'cc-pvtz',
    'cc-pcvdz': 'cc-pvdz',
}