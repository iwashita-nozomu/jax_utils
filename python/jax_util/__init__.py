__version__ = "0.1.0"

__all__ = ["__version__"]


if __name__ == "__main__":
	def test_version() -> None:
		assert isinstance(__version__, str)
		assert __version__

	test_version()
