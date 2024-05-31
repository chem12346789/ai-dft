from cadft import train_model

TRAIN_STR_DICT = [
    "Methane",
    "Ethane",
    "Ethylene",
    "Acetylene",
    "Allene",
    "Propane",
    "Propyne",
    "Cyclopropene",
    "Cyclopropane",
    "Propylene",
]
EVAL_STR_DICT = [
    "Isobutane",
    "Butane",
    "Butadiene",
    "Butyne",
    "Bicyclobutane",
    "Cyclobutane",
    "Spiropentane",
    "Cyclopropylmethyl",
    "Neopentane",
    "Cyclopentane",
    "Benzene",
    "Pentane",
    "Isopentane",
]

if __name__ == "__main__":
    train_model(TRAIN_STR_DICT, EVAL_STR_DICT)
