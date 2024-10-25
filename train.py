from cadft import train_model

TRAIN_STR_DICT = [
    "methane",
    "ethane",
    "ethylene",
    "acetylene",
    "cyclopropene",
    "cyclopropane",
    "methyl-openshell",
]
EVAL_STR_DICT = [
    "propane",
    # "allene",
    # "propyne",
    # "propylene",
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

