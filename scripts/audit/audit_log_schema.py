"""
Audit Log Schema — 監査ログデータモデル

JSON Schema と TypeScript 型定義で監査ログの形式を厳密に定義。
静的型チェック・バリデーション・実行時チェックで一貫性を保証。
"""

import json
from typing import Any, Dict, Optional, Literal
from dataclasses import dataclass, field, asdict
from datetime import datetime


# ========== JSON Schema Definition ==========

AUDIT_LOG_JSON_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "title": "Audit Log Entry",
    "description": "統一監査ログスキーマ",
    "type": "object",
    "required": [
        "timestamp",
        "action",
        "actor",
        "level",
        "outcome",
    ],
    "properties": {
        "timestamp": {
            "type": "string",
            "format": "date-time",
            "description": "ISO 8601 UTC timestamp"
        },
        "action": {
            "type": "string",
            "enum": [
                "skill_executed",
                "skill_failed",
                "code_reviewed",
                "test_run",
                "experiment_started",
                "experiment_completed",
                "security_check",
                "rbac_enforced",
                "secret_accessed",
                "deployment_started",
                "deployment_completed",
                "pr_opened",
                "pr_approved",
                "pr_merged",
            ],
            "description": "実行されたアクション"
        },
        "actor": {
            "type": "string",
            "description": "実行者（ユーザー/ロール/システムコンポーネント）"
        },
        "level": {
            "type": "string",
            "enum": ["INFO", "WARNING", "ERROR", "SECURITY", "COMPLIANCE"],
            "description": "ログレベル"
        },
        "resource": {
            "type": ["string", "null"],
            "description": "対象リソース（ファイル/PR/etc）"
        },
        "outcome": {
            "type": "string",
            "enum": ["success", "failure", "warning", "partial"],
            "description": "実行結果"
        },
        "details": {
            "type": "object",
            "description": "アクション固有の詳細情報",
            "additionalProperties": True
        },
        "metadata": {
            "type": "object",
            "description": "追加メタデータ",
            "properties": {
                "duration_ms": {
                    "type": "integer",
                    "description": "実行時間（ミリ秒）"
                },
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "タグリスト"
                }
            },
            "additionalProperties": True
        },
        "git_commit": {
            "type": "string",
            "description": "Git commit SHA（最初の8文字）"
        },
        "branch": {
            "type": "string",
            "description": "Git branch name"
        },
        "error": {
            "type": ["object", "null"],
            "properties": {
                "type": {"type": "string"},
                "message": {"type": "string"},
                "traceback": {"type": ["string", "null"]},
            },
            "description": "エラー情報（失敗時）"
        }
    },
    "additionalProperties": False
}


# ========== TypeScript Type Definitions (as comments) ==========

TYPE_DEFINITIONS = '''
// TypeScript definitions for audit log (reference)

type AuditLevel = "INFO" | "WARNING" | "ERROR" | "SECURITY" | "COMPLIANCE";

type AuditAction = 
  | "skill_executed"
  | "skill_failed"
  | "code_reviewed"
  | "test_run"
  | "experiment_started"
  | "experiment_completed"
  | "security_check"
  | "rbac_enforced"
  | "secret_accessed"
  | "deployment_started"
  | "deployment_completed"
  | "pr_opened"
  | "pr_approved"
  | "pr_merged";

type AuditOutcome = "success" | "failure" | "warning" | "partial";

interface AuditLogEntry {
  timestamp: string;          // ISO 8601 UTC
  action: AuditAction;
  actor: string;              // user/role/system component
  level: AuditLevel;
  resource?: string | null;   // file/PR/resource identifier
  outcome: AuditOutcome;
  details: Record<string, any>;
  metadata?: Record<string, any> & {
    duration_ms?: number;
    tags?: string[];
  };
  git_commit: string;         // first 8 chars of SHA
  branch: string;
  error?: {
    type: string;
    message: string;
    traceback?: string | null;
  } | null;
}

interface AuditLogQuery {
  action?: AuditAction;
  actor?: string;
  level?: AuditLevel;
  outcome?: AuditOutcome;
  resource?: string;
  start_date?: string;        // ISO 8601
  end_date?: string;          // ISO 8601
  limit?: number;
}

interface AuditLogStatistics {
  total_entries: number;
  date_range: {
    start: string;
    end: string;
  };
  by_action: Record<AuditAction, number>;
  by_actor: Record<string, number>;
  by_level: Record<AuditLevel, number>;
  by_outcome: Record<AuditOutcome, number>;
  error_rate: number;         // percentage
  security_events: number;
}
'''


# ========== Python Dataclass Models ==========

@dataclass
class ErrorInfo:
    """エラー情報"""
    type: str
    message: str
    traceback: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AuditLogMetadata:
    """監査ログメタデータ"""
    duration_ms: Optional[int] = None
    tags: list[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        return {k: v for k, v in data.items() if v is not None}


@dataclass
class AuditLogEntry:
    """監査ログエントリ"""
    timestamp: str              # ISO 8601 UTC
    action: str
    actor: str
    level: str                  # INFO, WARNING, ERROR, SECURITY, COMPLIANCE
    outcome: str                # success, failure, warning, partial
    details: Dict[str, Any] = field(default_factory=dict)
    resource: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    git_commit: str = "unknown"
    branch: str = "unknown"
    error: Optional[Dict[str, Any]] = None
    
    def validate(self) -> bool:
        """スキーマバリデーション"""
        import jsonschema
        
        schema_validator = jsonschema.Draft7Validator(AUDIT_LOG_JSON_SCHEMA)
        errors = list(schema_validator.iter_errors(self.to_dict()))
        
        if errors:
            for error in errors:
                print(f"Validation error: {error.message}")
            return False
        
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        """辞書に変換"""
        data = {
            "timestamp": self.timestamp,
            "action": self.action,
            "actor": self.actor,
            "level": self.level,
            "outcome": self.outcome,
            "details": self.details,
            "git_commit": self.git_commit,
            "branch": self.branch,
        }
        
        if self.resource:
            data["resource"] = self.resource
        
        if self.metadata:
            data["metadata"] = self.metadata
        
        if self.error:
            data["error"] = self.error
        
        return data
    
    def to_json(self) -> str:
        """JSON 文字列に変換"""
        return json.dumps(self.to_dict(), ensure_ascii=False)


@dataclass
class AuditLogQuery:
    """監査ログクエリ"""
    action: Optional[str] = None
    actor: Optional[str] = None
    level: Optional[str] = None
    outcome: Optional[str] = None
    resource: Optional[str] = None
    start_date: Optional[str] = None  # ISO 8601
    end_date: Optional[str] = None     # ISO 8601
    limit: int = 100
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        return {k: v for k, v in data.items() if v is not None}


@dataclass
class AuditLogStatistics:
    """監査ログ統計"""
    total_entries: int
    date_range: Dict[str, str]
    by_action: Dict[str, int]
    by_actor: Dict[str, int]
    by_level: Dict[str, int]
    by_outcome: Dict[str, int]
    error_rate: float
    security_events: int
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)


# ========== Schema Export ==========

def get_json_schema() -> Dict[str, Any]:
    """JSON Schema を取得"""
    return AUDIT_LOG_JSON_SCHEMA


def get_typescript_definitions() -> str:
    """TypeScript 型定義を取得"""
    return TYPE_DEFINITIONS


def validate_entry(entry: Dict[str, Any]) -> tuple[bool, Optional[str]]:
    """ログエントリをバリデーション
    
    Args:
        entry: バリデーション対象の辞書
    
    Returns:
        (是否, エラーメッセージ or None)
    """
    try:
        import jsonschema
        jsonschema.validate(entry, AUDIT_LOG_JSON_SCHEMA)
        return True, None
    except Exception as e:
        return False, str(e)


if __name__ == "__main__":
    print("Audit Log Schema")
    print("=" * 60)
    
    print("\n1. JSON Schema:")
    print(json.dumps(get_json_schema(), indent=2, ensure_ascii=False)[:500])
    
    print("\n2. TypeScript Definitions:")
    print(get_typescript_definitions()[:300])
    
    # テストエントリ
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
    
    print("\n3. Sample Entry:")
    print(entry.to_json())
    
    # バリデーション
    is_valid, error = validate_entry(entry.to_dict())
    print(f"\n4. Validation: {'✅ PASS' if is_valid else '❌ FAIL'}")
    if error:
        print(f"   Error: {error}")
