from .LGBM import LightGBM
from .LR import LogisticRegression
from .RF import RandomForest
integrator_abb_map = {
    "LR"    : LogisticRegression,
    "RF"    : RandomForest,
    "LGBM" : LightGBM
}