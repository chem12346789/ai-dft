from cadft import train_model

TRAIN_STR_DICT = [
    "methane",
    "ethane",
    "ethylene",
    "acetylene",
]
EVAL_STR_DICT = [
    "propane",
    # "propyne",
    # "propylene",
    # "cyclopropene",
    # "cyclopropane",
    # "allene",
    # "butane",
    # "isobutane",
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
