#動作確認系

import os 
os.environ["JAX_LOG_COMPILES"] = "1"

import jax
from typing import Callable


def foo(x:jax.Array,y:jax.Array)->jax.Array:
    return x + y


def bar(f: Callable[[jax.Array],jax.Array],x:jax.Array)-> jax.Array:
    return f(x)
@jax.jit
def run(x: jax.Array,y:jax.Array)-> None:

    
    for i in range(10):
        foox = lambda x :foo(x,y)
        z = bar(foox,x)
        y= z
        jax.debug.print("{z}", z = z)
if __name__ == "__main__":

    x = jax.numpy.array([1.0,2.0,3.0])
    y = jax.numpy.array([4.0,5.0,6.0])
    run(x,y)

    u = jax.numpy.array([6.0,6.0,6.0])
    v = jax.numpy.array([4.0,5.0,6.0])
    run(u,v)