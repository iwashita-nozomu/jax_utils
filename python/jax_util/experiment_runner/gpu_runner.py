from __future__ import annotations

# GPU ベースの runner 実装。
# 将来、GPU リソース管理が必要になった場合にこのファイルを拡張する予定です。
# 現在は標準実装（cpu ベース）を使う方針です。

from .protocols import *
from .runner import *

