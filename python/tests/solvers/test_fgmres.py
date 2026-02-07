from __future__ import annotations


def test_fgmres_archived() -> None:
    """FGMRES はアーカイブ済みのため使用しません。"""
    assert True


def _run_all_tests() -> None:
    """このファイル内のテストを順に実行します。"""
    test_fgmres_archived()


if __name__ == "__main__":
    _run_all_tests()