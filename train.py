from cadft import train_model

TRAIN_STR_DICT = [
    "methane",
    # "methyl-openshell",
    "ethane",
    # "ethylene",
    # "acetylene",
    # "allene",
    # "cyclopropene",
    # "cyclopropane",
]
EVAL_STR_DICT = [
    "propane",
    # "propylene",
    # "propyne",
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
