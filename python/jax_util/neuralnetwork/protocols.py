from typing import Protocol


from base import *

class NNlayer(Operator,Protocol):
    weights: Matrix
    biases: Vector

