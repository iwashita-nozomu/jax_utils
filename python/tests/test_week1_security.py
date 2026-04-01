"""
Week 1 Security Integration Tests

監査ログ・RBAC・秘密管理の統合テスト
"""

import sys
import json
from pathlib import Path
from datetime import datetime, timedelta

# パスを設定
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "python"))
sys.path.insert(0, str(Path(__file__).parent.parent / "security"))
sys.path.insert(0, str(Path(__file__).parent.parent / "audit"))

# インポート
from scripts.audit.audit_logger import AuditLogger, AuditLevel
from scripts.audit.audit_log_schema import (
    AuditLogEntry,
    validate_entry,
    AuditLogStatistics,
)
from scripts.security.rbac_manager import RBACManager, RoleLevel
from scripts.security.secrets_vault import SecretsVault, SecretType


test_results = {
    "passed": 0,
    "failed": 0,
    "tests": [],
}


def test_audit_logger():
    """監査ロガーテスト"""
    print("\n📋 Testing Audit Logger...")
    
    logger = AuditLogger()
    
    # Test 1: ログ記録
    entry = logger.log(
        action="test_action",
        actor="test_user",
        level=AuditLevel.INFO,
        details={"test": "data"},
    )
    
    assert entry["action"] == "test_action"
    assert entry["actor"] == "test_user"
    print("  ✅ Log entry created")
    test_results["passed"] += 1
    
    # Test 2: ログ検索
    logs = logger.get_logs(actor="test_user", limit=10)
    assert len(logs) > 0
    print(f"  ✅ Found {len(logs)} log entries")
    test_results["passed"] += 1
    
    # Test 3: 統計
    stats = logger.get_statistics()
    assert stats["total_entries"] > 0
    print(f"  ✅ Statistics: {stats['total_entries']} total entries")
    test_results["passed"] += 1


def test_rbac_manager():
    """RBAC マネージャーテスト"""
    print("\n🔐 Testing RBAC Manager...")
    
    manager = RBACManager()
    
    # Test 1: ユーザー追加
    manager.add_user("test_user", ["developer"], "internal")
    assert "test_user" in manager.users
    print("  ✅ User added")
    test_results["passed"] += 1
    
    # Test 2: 権限チェック
    has_execute = manager.check_permission("test_user", "skill", "execute")
    assert has_execute is True
    print("  ✅ Developer can execute skills")
    test_results["passed"] += 1
    
    has_delete = manager.check_permission("test_user", "skill", "delete")
    assert has_delete is False
    print("  ✅ Developer cannot delete skills")
    test_results["passed"] += 1
    
    # Test 3: 権限取得
    perms = manager.get_permissions("test_user")
    assert "skill" in perms
    assert "execute" in perms["skill"]
    print(f"  ✅ Got permissions: {list(perms.keys())}")
    test_results["passed"] += 1


def test_secrets_vault():
    """シークレットボルトテスト"""
    print("\n🔑 Testing Secrets Vault...")
    
    vault = SecretsVault()
    
    # Test 1: シークレット設定
    secret = vault.set_secret(
        "test_secret",
        "super_secret_value_123",
        secret_type="token",
        tags=["test"],
    )
    assert secret.name == "test_secret"
    print("  ✅ Secret stored")
    test_results["passed"] += 1
    
    # Test 2: シークレット取得
    retrieved = vault.get_secret("test_secret")
    assert retrieved is not None
    assert retrieved.value == "super_secret_value_123"
    print("  ✅ Secret retrieved")
    test_results["passed"] += 1
    
    # Test 3: リスト（マスク）
    secrets = vault.list_secrets()
    assert len(secrets) > 0
    masked_value = secrets[0]["value"]
    assert "****" in masked_value or masked_value.endswith("***")
    print("  ✅ Secrets listed (values masked)")
    test_results["passed"] += 1
    
    # Test 4: 検証
    stats = vault.validate_secrets()
    assert stats["total"] > 0
    print(f"  ✅ Validation: {stats['total']} secrets")
    test_results["passed"] += 1


def test_audit_log_schema():
    """監査ログスキーマバリデーション"""
    print("\n📝 Testing Audit Log Schema...")
    
    # Test 1: スキーマバリデーション
    entry = AuditLogEntry(
        timestamp=datetime.utcnow().isoformat() + "Z",
        action="skill_executed",
        actor="test_user",
        level="INFO",
        outcome="success",
        details={"skill_id": "01-static-check"},
        git_commit="abc12345",
        branch="main",
    )
    
    is_valid, error = validate_entry(entry.to_dict())
    assert is_valid is True
    print("  ✅ Schema validation passed")
    test_results["passed"] += 1
    
    # Test 2: JSON 出力
    json_str = entry.to_json()
    parsed = json.loads(json_str)
    assert parsed["action"] == "skill_executed"
    print("  ✅ JSON serialization works")
    test_results["passed"] += 1


def test_integration():
    """統合テスト：監査ログ + RBAC + シークレット"""
    print("\n🔗 Testing Integration...")
    
    logger = AuditLogger()
    manager = RBACManager()
    vault = SecretsVault()
    
    # ユーザーを追加
    manager.add_user("integration_user", ["researcher"], "research")
    
    # シークレットを設定
    vault.set_secret(
        "integration_secret",
        "secret_value",
        secret_type="api_key",
    )
    
    # 権限チェックの結果をログ
    can_execute = manager.check_permission("integration_user", "skill", "execute")
    logger.log(
        action="permission_check",
        actor="integration_user",
        level=AuditLevel.INFO,
        outcome="success" if can_execute else "warning",
        details={
            "resource": "skill",
            "permission": "execute",
            "granted": can_execute,
        },
    )
    
    # 監査ログを検索
    logs = logger.get_logs(action="permission_check")
    assert len(logs) > 0
    
    print("  ✅ Audit + RBAC + Vault integrated")
    test_results["passed"] += 1


def main():
    """メイン テスト実行"""
    print("\n" + "=" * 60)
    print("Week 1 Security Integration Tests")
    print("=" * 60)
    
    try:
        test_audit_logger()
        test_rbac_manager()
        test_secrets_vault()
        test_audit_log_schema()
        test_integration()
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        test_results["failed"] += 1
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        test_results["failed"] += 1
    
    # 結果出力
    print("\n" + "=" * 60)
    print("Test Results")
    print("=" * 60)
    print(f"✅  Passed: {test_results['passed']}")
    print(f"❌  Failed: {test_results['failed']}")
    print(f"📊 Total:  {test_results['passed'] + test_results['failed']}")
    
    if test_results['failed'] == 0:
        print("\n🎉 All tests passed!")
        return 0
    else:
        print(f"\n⚠️  {test_results['failed']} test(s) failed")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
