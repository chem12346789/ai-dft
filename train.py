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
    # "butadiene",
    # "butyne",
    # "bicyclobutane",
    # "cyclobutane",
    # "benzene",
    # "spiropentane",
    # "cyclopropylmethyl",
    # "neopentane",
    # "cyclopentane",
    # "pentane",
    # "isopentane",
]

if __name__ == "__main__":
    train_model(TRAIN_STR_DICT, EVAL_STR_DICT)
