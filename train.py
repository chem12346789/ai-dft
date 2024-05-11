from cadft import train_model

ATOM_LIST = [
    "H",
    "C",
]
TRAIN_STR_DICT = [
    "Methane",
    "Ethane",
    "Ethylene",
    "Acetylene",
    # "Allene",
    # "Propane",
    # "Propyne",
    # "Cyclopropene",
    # "Cyclopropane",
    # "Propylene",
    # "Isobutane",
    # "Butane",
    # "Butadiene",
    # "Butyne",
    # "Bicyclobutane",
    # "Cyclopropylmethyl",
    # "Cyclobutane",
    # "Spiropentane",
    # "Benzene",
]
EVAL_STR_DICT = [
    "Propane",
    # "Pentane",
    # "Isopentane",
    # "Neopentane",
    # "Cyclopentane",
]

train_model(ATOM_LIST, TRAIN_STR_DICT, EVAL_STR_DICT)
