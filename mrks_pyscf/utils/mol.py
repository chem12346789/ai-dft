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

BASIS_PSI4 = {
    "cc-pcvtz": {
        "Be": """
Be     S
   6863.0000000              2.3600000e-04
   1030.0000000              1.8260000e-03
    234.7000000              9.4520000e-03
     66.5600000              3.7957000e-02
     21.6900000              1.1996500e-01
      7.7340000              2.8216200e-01
      2.9160000              4.2740400e-01
      1.1300000              2.6627800e-01
      0.2577000              1.8193000e-02
      0.1101000             -7.2750000e-03
      0.0440900              1.9030000e-03
Be     S
   6863.0000000             -4.3000000e-05
   1030.0000000             -3.3300000e-04
    234.7000000             -1.7360000e-03
     66.5600000             -7.0120000e-03
     21.6900000             -2.3126000e-02
      7.7340000             -5.8138000e-02
      2.9160000             -1.1455600e-01
      1.1300000             -1.3590800e-01
      0.2577000              2.2802600e-01
      0.1101000              5.7744100e-01
      0.0440900              3.1787300e-01
Be     S
      0.2577000              1.0000000e+00
Be     S
      0.0440900              1.0000000e+00
Be     P
      7.4360000              1.0736000e-02
      1.5770000              6.2854000e-02
      0.4352000              2.4818000e-01
      0.1438000              5.2369900e-01
      0.0499400              3.5342500e-01
Be     P
      0.1438000              1.0000000e+00
Be     P
      0.0499400              1.0000000e+00
Be     D
      0.3493000              1.0000000e+00
Be     D
      0.1724000              1.0000000e+00
Be     F
      0.3423000              1.0000000e+00
Be     S
      4.722900E+00           1.000000E+00
Be     S
      1.645100E+00           1.000000E+00
Be     P
      1.455000E+01           1.000000E+00
Be     P
      3.797700E+00           1.000000E+00
Be     D
      9.184100E+00           1.000000E+00"""
    }
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
