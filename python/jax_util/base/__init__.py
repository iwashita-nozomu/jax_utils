from __future__ import annotations

if __name__ == "__main__":
    from protocols import *
    from linearoperator import *
    from _env_value import *

else:
    from .protocols import *
    from ._env_value import *  
    from .linearoperator import *


__all__ = [
    "DEFAULT_DTYPE",
    "EPS",
    "DEBUG",
    "ZERO",
    "ONE",
    "HALF",
    "WEAK_EPS",
    "AVOID_ZERO_DIV",
    "LinOp",
    "Scalar",
    "Vector",
    "Matrix",
    "Boolean",
    "Integer",
    "LinearOperator",
    "Operator",
]


if __name__ == "__main__":
    import jax.numpy as jnp

    def test_exports() -> None:
        s: Scalar = jnp.asarray(1.0)
        v: Vector = jnp.asarray([1.0, 2.0])
        m: Matrix = jnp.asarray([[1.0, 0.0], [0.0, 1.0]])
        assert s.shape == ()
        assert v.shape == (2,)
        assert m.shape == (2, 2)

    test_exports()
