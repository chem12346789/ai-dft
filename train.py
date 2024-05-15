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
    # "Benzene",
]
EVAL_STR_DICT = [
    "Pentane",
    # "Isopentane",
    # "Neopentane",
    # "Cyclopentane",
]

train_model(ATOM_LIST, TRAIN_STR_DICT, EVAL_STR_DICT)
