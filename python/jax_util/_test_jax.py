import jax

print("JAX version:", jax.__version__)
print(jax.devices())

from base import *


A :LinearOperator = jax.numpy.array([[1.,2.],[3.,4.]])


if __name__ == "__main__":
	def test_jax_available() -> None:
		assert jax.__version__
		assert len(jax.devices()) > 0

	def test_linear_operator_shape() -> None:
		v: Vector = jax.numpy.array([1.0, 1.0])
		out = A @ v
		assert out.shape == (2,)

	test_jax_available()
	test_linear_operator_shape()