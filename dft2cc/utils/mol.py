"""Molecular dict"""

H_2 = [["H", 100, 0, 0], ["H", 0, 0, 0]]
HHHH = [
    ["H", 100, 0, 0],
    ["H", 0, 0, 0],
    ["H", -1.0, 0, 0],
    ["H", -1.737, 0, 0],
]
Be = [["Be", 0, 0, 0]]
Ne = [["Ne", 0, 0, 0]]
He = [["He", 0, 0, 0]]
HF = [["H", 100, 0, 0], ["F", 0, 0, 0]]
HCN = [
    ["H", 100, 0, 0],
    ["C", 0, 0, 0],
    ["N", -1.1560, 0, 0],
]
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
HCHHH = [
    ["H", 1.1016, -0.0, 0.0],
    ["C", 0.0, 0.0, 0.0],
    ["H", -0.3672, -0.0, -1.0386],
    ["H", -0.3672, -0.8994, 0.5193],
    ["H", -0.0362, -0.8994, -0.635],
]

Mol = {
    "H_2": H_2,
    "HH": H_2,
    "HHHH": HHHH,
    "HF": HF,
    "Be": Be,
    "He": He,
    "Ne": Ne,
    "HCN": HCN,
    "HOH": HOH,
    "HOOH": HOOH,
    "HNHH": HNHH,
    "HNH2": HNHH,
    "HCHHH": HCHHH,
    "HCH3": HCHHH,
}

PREDICT_MOLECULAR = {
    "HHHH": "HH",
}

BASIS = {
    # aug-cc-pwcv
    "aug-cc-pwcv6z": "aug-cc-pv6z",
    "aug-cc-pwcv5z": "aug-cc-pv5z",
    "aug-cc-pwcvqz": "aug-cc-pvqz",
    "aug-cc-pwcvtz": "aug-cc-pvtz",
    "aug-cc-pwcvdz": "aug-cc-pvdz",
    # aug-cc-pwcv
    "augccpwcv6z": "aug-cc-pv6z",
    "augccpwcv5z": "aug-cc-pv5z",
    "augccpwcvqz": "aug-cc-pvqz",
    "augccpwcvtz": "aug-cc-pvtz",
    "augccpwcvdz": "aug-cc-pvdz",
    # aug-cc-pcv
    "aug-cc-pcv6z": "aug-cc-pv6z",
    "aug-cc-pcv5z": "aug-cc-pv5z",
    "aug-cc-pcvqz": "aug-cc-pvqz",
    "aug-cc-pcvtz": "aug-cc-pvtz",
    "aug-cc-pcvdz": "aug-cc-pvdz",
    # ccpwpcv
    "ccpwpcv6z": "cc-pv6z",
    "ccpwpcv5z": "cc-pv5z",
    "ccpwpcvqz": "cc-pvqz",
    "ccpwpcvtz": "cc-pvtz",
    "ccpwpcvdz": "cc-pvdz",
    # cc-pcv
    "cc-pcv6z": "cc-pv6z",
    "cc-pcv5z": "cc-pv5z",
    "cc-pcvqz": "cc-pvqz",
    "cc-pcvtz": "cc-pvtz",
    "cc-pcvdz": "cc-pvdz",
}

BASISTRAN = {
    "AhlrichspVDZ": "Ahlrichs pVDZ",
    "AhlrichsTZV": "Ahlrichs TZV",
    "AhlrichsVDZ": "Ahlrichs VDZ",
    "AhlrichsVTZ": "Ahlrichs VTZ",
}

MASS = {
    "H": 1.00782503207,
    "He": 4.00260325415,
    "Li": 6.938,
    "Be": 9.012183065,
    "B": 10.806,
    "C": 12.0096,
    "N": 14.006855,
    "O": 15.9994,
    "F": 18.998403163,
    "Ne": 20.1797,
}
