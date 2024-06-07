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
    "Butane",
    "Butadiene",
    "Butyne",
    "Isobutane",
    "Bicyclobutane",
    "Cyclobutane",
    "Benzene",
    "Spiropentane",
    "Cyclopropylmethyl",
    "Neopentane",
    "Cyclopentane",
]
EVAL_STR_DICT = [
    "Pentane",
    "Isopentane",
]

if __name__ == "__main__":
    train_model(TRAIN_STR_DICT, EVAL_STR_DICT)
