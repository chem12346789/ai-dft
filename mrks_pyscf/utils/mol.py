"""Molecular dict"""

H_2 = [["H", 100, 0, 0], ["H", 0, 0, 0]]
HHHH = [
    ["H", 100, 0, 0],
    ["H", 0, 0, 0],
    ["H", 1.0, 0, 0],
    ["H", 1.737, 0, 0],
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
HOH1 = [
    ["H", 1.0, 0.0, 0.0],
    ["O", 0.0, 0.0, 0.0],
    ["H", 0.0, 1.0, 0.0],
]
HOH2 = [
    ["H", 1.0, 0.0, 0.0],
    ["O", 0.0, 0.0, 0.0],
    ["H", -1.0, 0.0, 0.0],
]
HOH3 = [
    ["H", 1.0, 0.0, 0.0],
    ["O", 0.0, 0.0, 0.0],
    ["H", -0.18285, 0.72735, 0.0],
]
HOH4 = [
    ["H", 1.0, 0.0, 0.0],
    ["O", 0.0, 0.0, 0.0],
    ["H", -0.30475, 1.21225, 0.0],
]
HOH5 = [
    ["H", 1.0, 0.0, 0.0],
    ["O", 0.0, 0.0, 0.0],
    ["H", -0.19504, 0.77584, 0.0],
]
HOH6 = [
    ["H", 1.0, 0.0, 0.0],
    ["O", 0.0, 0.0, 0.0],
    ["H", -0.20723, 0.82433, 0.0],
]
HOH7 = [
    ["H", 1.0, 0.0, 0.0],
    ["O", 0.0, 0.0, 0.0],
    ["H", -0.21942, 0.87282, 0.0],
]
HOH8 = [
    ["H", 1.0, 0.0, 0.0],
    ["O", 0.0, 0.0, 0.0],
    ["H", -0.23161, 0.92131, 0.0],
]
HOH9 = [
    ["H", 1.0, 0.0, 0.0],
    ["O", 0.0, 0.0, 0.0],
    ["H", -0.25599, 1.01829, 0.0],
]
HOH10 = [
    ["H", 1.0, 0.0, 0.0],
    ["O", 0.0, 0.0, 0.0],
    ["H", -0.26818, 1.06678, 0.0],
]
HOH11 = [
    ["H", 1.0, 0.0, 0.0],
    ["O", 0.0, 0.0, 0.0],
    ["H", -0.28037, 1.11527, 0.0],
]
HOH12 = [
    ["H", 1.0, 0.0, 0.0],
    ["O", 0.0, 0.0, 0.0],
    ["H", -0.29256, 1.16376, 0.0],
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
    "HOH1": HOH1,
    "HOH2": HOH2,
    "HOH3": HOH3,
    "HOH4": HOH4,
    "HOH5": HOH5,
    "HOH6": HOH6,
    "HOH7": HOH7,
    "HOH8": HOH8,
    "HOH9": HOH9,
    "HOH10": HOH10,
    "HOH11": HOH11,
    "HOH12": HOH12,
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
    else:
        return 0.99


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
    else:
        return 0.99


def old_function3(distance):
    """
    This function is used to determine the factor of mixing old and new density matrix in SCF process
    """
    if distance < 1.5:
        return 0.95
    if distance < 2.5:
        return 0.975
    if distance < 3.5:
        return 0.99
    else:
        return 0.995


def old_function4(distance):
    """
    This function is used to determine the factor of mixing old and new density matrix in SCF process
    """
    if distance < 1.5:
        return 0.975
    if distance < 2.5:
        return 0.99
    if distance < 3.5:
        return 0.995
    else:
        return 0.999


def old_function5(distance):
    """
    This function is used to determine the factor of mixing old and new density matrix in SCF process
    """
    if distance < 1.5:
        return 0.99
    if distance < 2.5:
        return 0.995
    if distance < 3.5:
        return 0.999
    else:
        return 0.9999


def old_function(distance, old_factor_scheme, old_factor):
    if old_factor_scheme == 1:
        FRAC_OLD = old_function1(distance)
    elif old_factor_scheme == 2:
        FRAC_OLD = old_function2(distance)
    elif old_factor_scheme == 3:
        FRAC_OLD = old_function3(distance)
    elif old_factor_scheme == 4:
        FRAC_OLD = old_function4(distance)
    elif old_factor_scheme == 5:
        FRAC_OLD = old_function5(distance)
    else:
        FRAC_OLD = old_factor
    return FRAC_OLD
