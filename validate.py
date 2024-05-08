from cadft import validate_model

ATOM_LIST = [
    "H",
    "C",
]
TRAIN_STR_DICT = [
    # "Methane",
    "Ethane",
    # "Ethylene",
    # "Acetylene",
    # "Allene",
    # "Cyclopropene",
    # "Propyne",
    # "Cyclopropane",
    # "Propylene",
    # "Propane",
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
    # "Pentane",
    # "Isopentane",
    # "Neopentane",
    # "Cyclopentane",
]

validate_model(ATOM_LIST, TRAIN_STR_DICT, EVAL_STR_DICT)
