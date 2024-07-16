from cadft import train_model

TRAIN_STR_DICT = [
    "methane",
]
EVAL_STR_DICT = [
    "ethane",
    # "ethylene",
    # "acetylene",
    # "propane",
    # "propyne",
    # "propylene",
    # "allene",
    # "cyclopropene",
    # "cyclopropane",
    # "butane",
    # "butyne",
    # "isobutane",
    # "butadiene",
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
