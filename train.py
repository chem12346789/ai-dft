from cadft import train_model

ATOM_LIST = [
    "H",
    "C",
]
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
EVAL_STR_DICT = [
    "hexane",
    "Cyclopentane",
    "2-methylpentane",
    "3-methylpentane",
    "2,2-dimethylbutane",
    "2,3-dimethylbutane",
    "cyclohexane",
    "1-hexene",
    "methylcyclopentane",
    "3,3-dimethyl-1-butene",
    "4-methyl-1-pentene",
    "2-methyl-1-pentene",
    "2-hexene",
    "2,3-dimethyl-2-butene",
    "3-hexene",
    "2-ethyl-1-butene",
    "propylcyclopropane",
    "2,3-dimethyl-1-butene",
    "2-methyl-2-pentene",
    "4-methyl-2-pentene",
    "3-methyl-1-pentene",
    "2-pentene,3-methyl-",
    "1-ethyl-1-methylcyclopropane",
]

train_model(ATOM_LIST, TRAIN_STR_DICT, EVAL_STR_DICT)
