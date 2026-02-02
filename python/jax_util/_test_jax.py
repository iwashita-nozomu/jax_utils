import jax

print("JAX version:", jax.__version__)
print(jax.devices())

from base import *


A :LinearOperator = jax.numpy.array([[1.,2.],[3.,4.]])