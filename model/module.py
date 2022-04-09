import numpy as np
import scipy.sparse as ssp
from typing import (
    Union,
    Optional
)

Matrix = Union[np.ndarray,ssp.spmatrix]
class Module:
    def __init__(self,*args,**kwargs):
        self.training = False
        ...
    def train(self,*args,**kwargs):
        self.training =True
        ...
    def test(self,*args,**kwargs):
        self.training = False
        ...
    def __call__(self,*arg,**kwargs):
        return self.forward(*arg,**kwargs)

    def forward(self,*arg,**kwargs):
        ...
