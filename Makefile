.PHONY: git_init ci dev-setup tools-help

# Git 初期化（初回のみ）
git_init:
	bash scripts/git_init.sh
	@echo "✅ Git setup complete"

# ★推奨: 統合 CI（pytest + pyright + ruff）
ci:
	bash scripts/ci/run_all_checks.sh

# CI 高速モード（ruff skip）
ci-quick:
	bash scripts/ci/run_all_checks.sh --quick

# 開発環境初期化
dev-setup: git_init
	@echo "✅ Dev environment ready. Start with: bash scripts/setup_worktree.sh <branch-name>"

# ツール情報表示
tools-help:
	@echo "=== Available Tools & Scripts ==="
	@echo ""
	@cat scripts/README.md | head -40