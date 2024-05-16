from cadft import train_model

ATOM_LIST = [
    "H",
    "C",
]
TRAIN_STR_DICT = [
    "Methane",
    # "Ethane",
    # "Ethylene",
    # "Acetylene",
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
    # "Cyclobutane",
    # "Spiropentane",
    # "Cyclopropylmethyl",
    # "Neopentane",
    # "Cyclopentane",
    # "Benzene",
]
EVAL_STR_DICT = [
    "Ethane",
    # "Pentane",
    # "Isopentane",
]

train_model(ATOM_LIST, TRAIN_STR_DICT, EVAL_STR_DICT)
