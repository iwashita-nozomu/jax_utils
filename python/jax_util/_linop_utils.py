from _type_aliaces import *

import equinox as eqx

import jax

def ensure_batch(f:Map) -> BMap:
    """Ensure that a function is batched.

    If the function is already batched, it is returned as is.
    If not, it is wrapped to handle batched inputs.

    Args:
        f: A function that maps from Shaped[Array,"n"] to Shaped[Array,"m"].

    Returns:
        A function that maps from Shaped[Array,"b n"] to Shaped[Array,"b m"].
    """
    def batched_f(x:Shaped[Array,"b n"]) -> Shaped[Array,"b m"]:
        return jax.vmap(f)(x)
    
    return batched_f

class LinOp(eqx.Module,Generic[BLinearMap]):
    

    