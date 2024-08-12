from cadft import train_model

TRAIN_STR_DICT = [
    "methane",
    "methyl-openshell",
    # "ethane",
    # "ethylene",
    # "acetylene",
    # "cyclopropene",
    # "cyclopropane",
]
EVAL_STR_DICT = [
    "ethane",
    # "propane",
    # "propylene",
    # "allene",
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
