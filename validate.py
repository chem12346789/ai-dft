from cadft import validate_model

ATOM_LIST = [
    "H",
    "C",
]
TRAIN_STR_DICT = [
<<<<<<< HEAD
    "Methane",
    "Ethane",
    "Ethylene",
    "Acetylene",
    "Allene",
    "Cyclopropene",
    "Propyne",
    "Cyclopropane",
    "Propylene",
    "Propane",
    "Isobutane",
    "Butane",
    "Butadiene",
    "Butyne",
    "Bicyclobutane",
    "Cyclopropylmethyl",
    "Cyclobutane",
    "Spiropentane",
    "Benzene",
=======
    # "Methane",
    # "Ethane",
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
>>>>>>> 896dc6a96888e38f2cd9f12520515a3f665b55e6
]
EVAL_STR_DICT = [
    "Pentane",
    "Isopentane",
    "Neopentane",
    "Cyclopentane",
    "Pentane",
    # "Hexane",
]

validate_model(ATOM_LIST, TRAIN_STR_DICT, EVAL_STR_DICT)
