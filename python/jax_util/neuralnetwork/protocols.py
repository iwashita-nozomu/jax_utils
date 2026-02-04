from typing import Protocol

from typing import Tuple
from base import *

class Ctx(Protocol):#固定状態
    ...
class Carry(Protocol):#層の計算に使う状態zとか
    ...

class NeuralNetworkLayer(Protocol):
    def __call__(self, carry: Carry,ctx: Ctx,/) -> Tuple[Carry, Ctx]: ... # 順伝播
    def __matmal__(self,other:"NeuralNetworkLayer",/)->"NeuralNetworkLayer":...  # レイヤーの合成
    ...

__all__ = [
    "NeuralNetworkLayer",
]