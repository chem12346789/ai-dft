from cadft import train_model

TRAIN_STR_DICT = [
    "Methane",
    "Ethane",
    "Ethylene",
    "Acetylene",
    "Propane",
    "Propyne",
    "Allene",
    "Cyclopropene",
    "Cyclopropane",
    "Propylene",
]
EVAL_STR_DICT = [
    "Butane",
    "Isobutane",
    # "Butadiene",
    # "Butyne",
    # "Bicyclobutane",
    # "Cyclobutane",
    # "Benzene",
    # "Spiropentane",
    # "Cyclopropylmethyl",
    # "Neopentane",
    # "Cyclopentane",
    # "Pentane",
    # "Isopentane",
]

if __name__ == "__main__":
    train_model(TRAIN_STR_DICT, EVAL_STR_DICT)
