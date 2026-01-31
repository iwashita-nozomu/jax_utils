
from typing import TypeVar,Protocol,TypeAlias,Generic
from jaxtyping import Array,Shaped,jaxtyped
from typeguard import typechecked

Scalar : TypeAlias = Shaped [Array,""]

Vector: TypeAlias = Shaped [Array,"n"]
BatchedVector : TypeAlias = Shaped [Array,"b n"]

Matrix : TypeAlias = Shaped [Array,"m n"]
BatchedMatrix : TypeAlias = Shaped [Array,"b m n"]


#自己写像
A = TypeVar("A")

class Endo(Protocol[A]):
    def __call__(self,x:A) -> A: ...

#射
B = TypeVar("B",contravariant=True)
C = TypeVar("C",covariant=True)
class Hom(Protocol[B,C]):
    def __call__(self,x:B) -> C: ...

#写像
class Map(Hom[Shaped[Array,"n"],Shaped[Array,"m"]],Protocol):
    ...
    
class BMap(Hom[Shaped[Array,"b n"],Shaped[Array,"b m"]],Protocol):
    ...

class LinearMap(Map,Protocol):
    ...

class BLinearMap(BMap,Protocol):
    ...

if __name__ == "__main__":
    m=10
    def f(x:Shaped[Array,"n"]) -> Shaped[Array,"m"]:
        return x[..., :m]
    
    g:BMap = f

    