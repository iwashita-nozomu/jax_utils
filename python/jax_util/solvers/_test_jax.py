import json
from pathlib import Path

import jax

from ..base import LinearOperator

SOURCE_FILE = Path(__file__).name


# 責務: 実行中の JAX バージョンと利用可能デバイスを JSON で出力します。
def _print_jax_env() -> None:
	"""JAX の環境情報を標準出力へ JSON で出力します。"""
	print(
		json.dumps(
			{
				"case": "jax_env",
				"source_file": SOURCE_FILE,
				"event": "version",
				"jax_version": jax.__version__,
			}
		)
	)
	print(
		json.dumps(
			{
				"case": "jax_env",
				"source_file": SOURCE_FILE,
				"event": "devices",
				"devices": [str(device) for device in jax.devices()],
			}
		)
	)


# 責務: JAX の最小疎通確認だけを行います。
def _smoke_test() -> None:
	"""最小の smoke test を実行します。"""
	_ = jax.numpy.array([[1.0, 2.0], [3.0, 4.0]])
	_ = LinearOperator


if __name__ == "__main__":
	_print_jax_env()
	_smoke_test()
