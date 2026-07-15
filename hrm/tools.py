from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import tempfile
import time
import traceback
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urljoin, urlparse

try:
    import httpx  # type: ignore
except ImportError:  # pragma: no cover
    httpx = None

try:
    import jsonschema  # type: ignore
except ImportError:  # pragma: no cover
    jsonschema = None


class ToolEffect(Enum):
    READ_ONLY = "read_only"
    STATE_CHANGING = "state_changing"
    DESTRUCTIVE = "destructive"
    CODE_EXECUTING = "code_executing"
    NETWORKED = "networked"


class ToolRisk(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    input_schema: dict[str, Any]
    output_schema: dict[str, Any]
    effect: ToolEffect
    risk: ToolRisk
    required_permissions: frozenset[str]
    timeout_seconds: float
    network_access: bool
    executable_access: bool


@dataclass(frozen=True)
class PermissionContext:
    principal_id: str
    granted_permissions: frozenset[str]
    allowed_directories: tuple[Path, ...]
    allowed_hosts: tuple[str, ...]
    session_id: str
    expires_at: float | None = None
    valid: bool = True


@dataclass(frozen=True)
class VerificationResult:
    passed: bool
    checks: dict[str, bool]
    expected: dict[str, Any]
    observed: dict[str, Any]
    errors: tuple[str, ...]


@dataclass(frozen=True)
class ToolResult:
    tool_name: str
    input_data: Any
    success: bool
    output: str
    error: str | None = None
    verification: VerificationResult | None = None


@dataclass(frozen=True)
class AuditRecord:
    event_id: str
    timestamp: float
    session_id: str
    principal_id: str
    tool_name: str
    effect: str
    risk: str
    arguments: dict[str, Any]
    redacted_arguments: dict[str, Any]
    permission_decision: str
    start_time: float
    end_time: float
    result: dict[str, Any]
    verification: dict[str, Any] | None
    error: str | None
    pre_action: dict[str, Any]
    post_action: dict[str, Any]
    previous_hash: str | None
    record_hash: str


SENSITIVE_KEY_PATTERNS = re.compile(
    r"(api|secret|token|password|passwd|auth|credential|session|cookie|private_key|privatekey|access_key|authorization)",
    re.IGNORECASE,
)


def redact_sensitive_data(payload: Any, sensitive_keys: frozenset[str] | None = None) -> Any:
    sensitive_keys = sensitive_keys or frozenset({"authorization", "api_key", "password", "token", "secret", "session", "cookie", "private_key", "access_token"})
    if isinstance(payload, dict):
        redacted = {}
        for key, value in payload.items():
            if key.lower() in sensitive_keys or SENSITIVE_KEY_PATTERNS.search(key):
                redacted[key] = "<REDACTED>"
            else:
                redacted[key] = redact_sensitive_data(value, sensitive_keys)
        return redacted
    if isinstance(payload, list):
        return [redact_sensitive_data(item, sensitive_keys) for item in payload]
    if isinstance(payload, str):
        if any(pattern in payload.lower() for pattern in sensitive_keys):
            return "<REDACTED>"
        if re.search(r"(Bearer\s+[A-Za-z0-9\-\._~\+/]+=*)", payload):
            return re.sub(r"(Bearer\s+)[A-Za-z0-9\-\._~\+/]+=*", r"\1<REDACTED>", payload)
    return payload


class PathPolicy:
    def __init__(self, read_roots: list[Path] | None = None, write_roots: list[Path] | None = None) -> None:
        self.read_roots = [root.resolve() for root in (read_roots or [])]
        self.write_roots = [root.resolve() for root in (write_roots or [])]

    def _normalize_requested_path(self, requested_path: str) -> str:
        if "%" in requested_path:
            decoded = os.path.normpath(os.path.expanduser(requested_path))
            if "%2e" in requested_path.lower() or "%2f" in requested_path.lower():
                raise ValueError("Encoded traversal is not allowed")
            requested_path = decoded
        if ".." in requested_path or "../" in requested_path or "..\\" in requested_path:
            raise ValueError("Parent directory traversal is not allowed")
        if requested_path.startswith("~"):
            raise ValueError("Home directory expansion is not allowed")
        return requested_path

    def _resolve_path(self, requested_path: str, roots: list[Path]) -> Path:
        requested_path = self._normalize_requested_path(requested_path)
        candidate = Path(requested_path)
        if candidate.is_absolute():
            candidate = candidate.resolve(strict=False)
            for root in roots:
                try:
                    if candidate.is_relative_to(root):
                        return candidate
                except AttributeError:
                    if root in candidate.parents or candidate == root:
                        return candidate
            raise ValueError(f"Path {requested_path} is outside allowed roots")
        for root in roots:
            candidate_path = (root / candidate).resolve(strict=False)
            try:
                if candidate_path.is_relative_to(root):
                    return candidate_path
            except AttributeError:
                if root in candidate_path.parents or candidate_path == root:
                    return candidate_path
        raise ValueError(f"Relative path {requested_path} is not inside any allowed root")

    def resolve_read_path(self, requested_path: str, context: PermissionContext) -> Path:
        if not self.read_roots and not context.allowed_directories:
            raise PermissionError("No read roots configured")
        roots = self.read_roots or list(context.allowed_directories)
        return self._resolve_path(requested_path, roots)

    def resolve_write_path(self, requested_path: str, context: PermissionContext) -> Path:
        if not self.write_roots and not context.allowed_directories:
            raise PermissionError("No write roots configured")
        roots = self.write_roots or list(context.allowed_directories)
        return self._resolve_path(requested_path, roots)


def _compute_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


class AuditLogger:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.write_text("", encoding="utf-8")

    def _read_previous_hash(self) -> str | None:
        previous_hash: str | None = None
        with self.path.open("r", encoding="utf-8") as handle:
            for line in handle:
                try:
                    payload = json.loads(line)
                    previous_hash = payload.get("record_hash")
                except Exception:
                    previous_hash = None
        return previous_hash

    def append(self, record: AuditRecord) -> None:
        lines = []
        if self.path.exists():
            with self.path.open("r", encoding="utf-8") as handle:
                lines = [line.rstrip("\n") for line in handle if line.strip()]
        previous_hash = self._read_previous_hash()
        record_dict = {**record.__dict__, "previous_hash": previous_hash}
        content = json.dumps(record_dict, sort_keys=True)
        record_hash = _compute_hash((previous_hash or "") + content)
        final_record = {**record_dict, "record_hash": record_hash}
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(final_record) + "\n")

    def verify_chain(self) -> bool:
        previous_hash = ""
        with self.path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                record_hash = record.get("record_hash")
                if record.get("previous_hash") != (previous_hash or None):
                    return False
                content = {k: v for k, v in record.items() if k not in {"record_hash", "previous_hash"}}
                canonical = json.dumps(content, sort_keys=True)
                expected_hash = _compute_hash((previous_hash or "") + canonical)
                if expected_hash != record_hash:
                    return False
                previous_hash = record_hash
        return True


class Tool:
    def __init__(self, spec: ToolSpec, executor: Callable[[dict[str, Any], PermissionContext, PathPolicy], ToolResult]) -> None:
        self.spec = spec
        self.name = spec.name
        self.description = spec.description
        self.executor = executor


class ToolRegistry:
    def __init__(self) -> None:
        self.tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        if not tool.spec.required_permissions:
            raise ValueError("Tool must declare required permissions")
        self.tools[tool.name] = tool

    def lookup(self, tool_name: str) -> Tool:
        if tool_name not in self.tools:
            raise KeyError(f"Tool '{tool_name}' is not registered")
        return self.tools[tool_name]

    def list_tools(self) -> list[str]:
        return sorted(self.tools.keys())

    def register_builtin_tools(self, policy: PathPolicy | None = None) -> None:
        policy = policy or PathPolicy()
        self.register(Tool(
            ToolSpec(
                name="list_directory",
                description="List files inside an allowed directory.",
                input_schema={"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"], "additionalProperties": False},
                output_schema={"type": "object", "properties": {"files": {"type": "array", "items": {"type": "string"}}}, "required": ["files"], "additionalProperties": False},
                effect=ToolEffect.READ_ONLY,
                risk=ToolRisk.LOW,
                required_permissions=frozenset({"tool.read"}),
                timeout_seconds=2.0,
                network_access=False,
                executable_access=False,
            ),
            self._list_directory,
        ))
        self.register(Tool(
            ToolSpec(
                name="read_file",
                description="Read a text file from an allowed directory.",
                input_schema={"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"], "additionalProperties": False},
                output_schema={"type": "object", "properties": {"content": {"type": "string"}}, "required": ["content"], "additionalProperties": False},
                effect=ToolEffect.READ_ONLY,
                risk=ToolRisk.LOW,
                required_permissions=frozenset({"tool.read"}),
                timeout_seconds=2.0,
                network_access=False,
                executable_access=False,
            ),
            self._read_file,
        ))
        self.register(Tool(
            ToolSpec(
                name="hash_file",
                description="Compute the hash of a file in an allowed directory.",
                input_schema={"type": "object", "properties": {"path": {"type": "string"}, "algorithm": {"type": "string", "enum": ["sha256"]}}, "required": ["path"], "additionalProperties": False},
                output_schema={"type": "object", "properties": {"hash": {"type": "string"}}, "required": ["hash"], "additionalProperties": False},
                effect=ToolEffect.READ_ONLY,
                risk=ToolRisk.LOW,
                required_permissions=frozenset({"tool.read"}),
                timeout_seconds=2.0,
                network_access=False,
                executable_access=False,
            ),
            self._hash_file,
        ))
        self.register(Tool(
            ToolSpec(
                name="create_file",
                description="Create or overwrite a file in an allowed directory.",
                input_schema={"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"], "additionalProperties": False},
                output_schema={"type": "object", "properties": {"path": {"type": "string"}, "status": {"type": "string"}}, "required": ["path", "status"], "additionalProperties": False},
                effect=ToolEffect.STATE_CHANGING,
                risk=ToolRisk.MEDIUM,
                required_permissions=frozenset({"tool.write"}),
                timeout_seconds=3.0,
                network_access=False,
                executable_access=False,
            ),
            self._create_file,
        ))
        self.register(Tool(
            ToolSpec(
                name="write_file",
                description="Write content to an allowed file path atomically.",
                input_schema={"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"], "additionalProperties": False},
                output_schema={"type": "object", "properties": {"path": {"type": "string"}, "status": {"type": "string"}}, "required": ["path", "status"], "additionalProperties": False},
                effect=ToolEffect.STATE_CHANGING,
                risk=ToolRisk.MEDIUM,
                required_permissions=frozenset({"tool.write"}),
                timeout_seconds=3.0,
                network_access=False,
                executable_access=False,
            ),
            self._write_file,
        ))
        self.register(Tool(
            ToolSpec(
                name="append_file",
                description="Append text to an allowed file path.",
                input_schema={"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"], "additionalProperties": False},
                output_schema={"type": "object", "properties": {"path": {"type": "string"}, "status": {"type": "string"}}, "required": ["path", "status"], "additionalProperties": False},
                effect=ToolEffect.STATE_CHANGING,
                risk=ToolRisk.MEDIUM,
                required_permissions=frozenset({"tool.write"}),
                timeout_seconds=3.0,
                network_access=False,
                executable_access=False,
            ),
            self._append_file,
        ))
        self.register(Tool(
            ToolSpec(
                name="copy_file",
                description="Copy a file within allowed directories.",
                input_schema={"type": "object", "properties": {"source_path": {"type": "string"}, "destination_path": {"type": "string"}}, "required": ["source_path", "destination_path"], "additionalProperties": False},
                output_schema={"type": "object", "properties": {"source_path": {"type": "string"}, "destination_path": {"type": "string"}, "status": {"type": "string"}}, "required": ["source_path", "destination_path", "status"], "additionalProperties": False},
                effect=ToolEffect.STATE_CHANGING,
                risk=ToolRisk.MEDIUM,
                required_permissions=frozenset({"tool.write"}),
                timeout_seconds=3.0,
                network_access=False,
                executable_access=False,
            ),
            self._copy_file,
        ))
        self.register(Tool(
            ToolSpec(
                name="move_file",
                description="Move or rename a file within allowed directories.",
                input_schema={"type": "object", "properties": {"source_path": {"type": "string"}, "destination_path": {"type": "string"}}, "required": ["source_path", "destination_path"], "additionalProperties": False},
                output_schema={"type": "object", "properties": {"source_path": {"type": "string"}, "destination_path": {"type": "string"}, "status": {"type": "string"}}, "required": ["source_path", "destination_path", "status"], "additionalProperties": False},
                effect=ToolEffect.DESTRUCTIVE,
                risk=ToolRisk.HIGH,
                required_permissions=frozenset({"tool.delete"}),
                timeout_seconds=3.0,
                network_access=False,
                executable_access=False,
            ),
            self._move_file,
        ))
        self.register(Tool(
            ToolSpec(
                name="delete_file",
                description="Delete an allowed file path.",
                input_schema={"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"], "additionalProperties": False},
                output_schema={"type": "object", "properties": {"path": {"type": "string"}, "status": {"type": "string"}}, "required": ["path", "status"], "additionalProperties": False},
                effect=ToolEffect.DESTRUCTIVE,
                risk=ToolRisk.HIGH,
                required_permissions=frozenset({"tool.delete"}),
                timeout_seconds=3.0,
                network_access=False,
                executable_access=False,
            ),
            self._delete_file,
        ))
        self.register(Tool(
            ToolSpec(
                name="execute_code",
                description="Execute Python code in an isolated subprocess.",
                input_schema={"type": "object", "properties": {"code": {"type": "string"}, "timeout_seconds": {"type": "number"}, "memory_limit_mb": {"type": "integer"}, "cpu_time_seconds": {"type": "integer"}}, "required": ["code"], "additionalProperties": False},
                output_schema={"type": "object", "properties": {"stdout": {"type": "string"}, "stderr": {"type": "string"}, "exit_code": {"type": "integer"}, "reason": {"type": "string"}}, "required": ["stdout", "stderr", "exit_code", "reason"], "additionalProperties": False},
                effect=ToolEffect.CODE_EXECUTING,
                risk=ToolRisk.CRITICAL,
                required_permissions=frozenset({"tool.execute"}),
                timeout_seconds=5.0,
                network_access=False,
                executable_access=True,
            ),
            self._execute_code,
        ))
        self.register(Tool(
            ToolSpec(
                name="api_request",
                description="Send an HTTP request to an allowed host.",
                input_schema={"type": "object", "properties": {"method": {"type": "string"}, "url": {"type": "string"}, "headers": {"type": "object"}, "body": {"type": ["object", "string", "null"]}}, "required": ["method", "url"], "additionalProperties": False},
                output_schema={"type": "object", "properties": {"status": {"type": "string"}, "body": {}}, "required": ["status", "body"], "additionalProperties": True},
                effect=ToolEffect.NETWORKED,
                risk=ToolRisk.MEDIUM,
                required_permissions=frozenset({"tool.network"}),
                timeout_seconds=5.0,
                network_access=True,
                executable_access=False,
            ),
            self._api_request,
        ))

    def _validate_input_schema(self, spec: ToolSpec, arguments: dict[str, Any]) -> bool:
        if jsonschema is None or not spec.input_schema:
            return True
        try:
            jsonschema.validate(instance=arguments, schema=spec.input_schema)
            return True
        except Exception:
            return False

    def _validate_output_schema(self, spec: ToolSpec, output: dict[str, Any]) -> bool:
        if jsonschema is None or not spec.output_schema:
            return True
        try:
            jsonschema.validate(instance=output, schema=spec.output_schema)
            return True
        except Exception:
            return False

    def _list_directory(self, arguments: dict[str, Any], context: PermissionContext, policy: PathPolicy) -> ToolResult:
        try:
            path = policy.resolve_read_path(arguments["path"], context)
            if not path.exists() or not path.is_dir():
                raise FileNotFoundError(f"Directory not found: {path}")
            items = sorted([entry.name for entry in path.iterdir()])
            output = json.dumps({"files": items})
            return ToolResult("list_directory", arguments, True, output)
        except Exception as exc:
            return ToolResult("list_directory", arguments, False, "", str(exc))

    def _read_file(self, arguments: dict[str, Any], context: PermissionContext, policy: PathPolicy) -> ToolResult:
        try:
            path = policy.resolve_read_path(arguments["path"], context)
            if not path.exists() or not path.is_file():
                raise FileNotFoundError(f"File not found: {path}")
            text = path.read_text(encoding="utf-8")
            output = json.dumps({"content": text})
            return ToolResult("read_file", arguments, True, output)
        except Exception as exc:
            return ToolResult("read_file", arguments, False, "", str(exc))

    def _hash_file(self, arguments: dict[str, Any], context: PermissionContext, policy: PathPolicy) -> ToolResult:
        try:
            algorithm = arguments.get("algorithm", "sha256")
            path = policy.resolve_read_path(arguments["path"], context)
            if not path.exists() or not path.is_file():
                raise FileNotFoundError(f"File not found: {path}")
            data = path.read_bytes()
            if algorithm != "sha256":
                raise ValueError("Only sha256 is supported")
            digest = hashlib.sha256(data).hexdigest()
            output = json.dumps({"hash": digest})
            return ToolResult("hash_file", arguments, True, output)
        except Exception as exc:
            return ToolResult("hash_file", arguments, False, "", str(exc))

    def _create_file(self, arguments: dict[str, Any], context: PermissionContext, policy: PathPolicy) -> ToolResult:
        try:
            path = policy.resolve_write_path(arguments["path"], context)
            path.parent.mkdir(parents=True, exist_ok=True)
            content = arguments["content"].encode("utf-8")
            temp_file = Path(tempfile.mkstemp(prefix="create_", suffix=".tmp", dir=path.parent)[1])
            temp_file.write_bytes(content)
            temp_file.replace(path)
            verification = self._verify_file(path, content)
            if not verification.passed:
                raise ValueError("File verification failed")
            output = json.dumps({"path": str(path), "status": "created"})
            return ToolResult("create_file", arguments, True, output, verification=verification)
        except Exception as exc:
            return ToolResult("create_file", arguments, False, "", str(exc))

    def _write_file(self, arguments: dict[str, Any], context: PermissionContext, policy: PathPolicy) -> ToolResult:
        try:
            path = policy.resolve_write_path(arguments["path"], context)
            path.parent.mkdir(parents=True, exist_ok=True)
            content = arguments["content"].encode("utf-8")
            temp_file = Path(tempfile.mkstemp(prefix="write_", suffix=".tmp", dir=path.parent)[1])
            temp_file.write_bytes(content)
            temp_file.replace(path)
            verification = self._verify_file(path, content)
            if not verification.passed:
                raise ValueError("File verification failed")
            output = json.dumps({"path": str(path), "status": "written"})
            return ToolResult("write_file", arguments, True, output, verification=verification)
        except Exception as exc:
            return ToolResult("write_file", arguments, False, "", str(exc))

    def _append_file(self, arguments: dict[str, Any], context: PermissionContext, policy: PathPolicy) -> ToolResult:
        try:
            path = policy.resolve_write_path(arguments["path"], context)
            path.parent.mkdir(parents=True, exist_ok=True)
            existing = path.read_bytes() if path.exists() else b""
            append_bytes = arguments["content"].encode("utf-8")
            path.write_bytes(existing + append_bytes)
            verification = self._verify_file(path, existing + append_bytes)
            if not verification.passed:
                raise ValueError("File verification failed")
            output = json.dumps({"path": str(path), "status": "appended"})
            return ToolResult("append_file", arguments, True, output, verification=verification)
        except Exception as exc:
            return ToolResult("append_file", arguments, False, "", str(exc))

    def _copy_file(self, arguments: dict[str, Any], context: PermissionContext, policy: PathPolicy) -> ToolResult:
        try:
            source = policy.resolve_read_path(arguments["source_path"], context)
            destination = policy.resolve_write_path(arguments["destination_path"], context)
            if not source.exists() or not source.is_file():
                raise FileNotFoundError(f"Source not found: {source}")
            destination.parent.mkdir(parents=True, exist_ok=True)
            data = source.read_bytes()
            temp_file = Path(tempfile.mkstemp(prefix="copy_", suffix=".tmp", dir=destination.parent)[1])
            temp_file.write_bytes(data)
            temp_file.replace(destination)
            verification = self._verify_file(destination, data)
            if not verification.passed:
                raise ValueError("File verification failed")
            output = json.dumps({"source_path": str(source), "destination_path": str(destination), "status": "copied"})
            return ToolResult("copy_file", arguments, True, output, verification=verification)
        except Exception as exc:
            return ToolResult("copy_file", arguments, False, "", str(exc))

    def _move_file(self, arguments: dict[str, Any], context: PermissionContext, policy: PathPolicy) -> ToolResult:
        try:
            source = policy.resolve_read_path(arguments["source_path"], context)
            destination = policy.resolve_write_path(arguments["destination_path"], context)
            if not source.exists() or not source.is_file():
                raise FileNotFoundError(f"Source not found: {source}")
            destination.parent.mkdir(parents=True, exist_ok=True)
            data = source.read_bytes()
            temp_file = Path(tempfile.mkstemp(prefix="move_", suffix=".tmp", dir=destination.parent)[1])
            temp_file.write_bytes(data)
            temp_file.replace(destination)
            source.unlink()
            verification = self._verify_file(destination, data)
            if not verification.passed or source.exists():
                raise ValueError("Move verification failed")
            output = json.dumps({"source_path": str(source), "destination_path": str(destination), "status": "moved"})
            return ToolResult("move_file", arguments, True, output, verification=verification)
        except Exception as exc:
            return ToolResult("move_file", arguments, False, "", str(exc))

    def _delete_file(self, arguments: dict[str, Any], context: PermissionContext, policy: PathPolicy) -> ToolResult:
        try:
            path = policy.resolve_write_path(arguments["path"], context)
            if not path.exists():
                raise FileNotFoundError(f"File not found: {path}")
            pre_hash = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else ""
            path.unlink()
            exists = path.exists()
            if exists:
                raise ValueError("Deletion verification failed")
            verification = VerificationResult(
                passed=True,
                checks={"deleted": True, "exists": False},
                expected={"exists": False, "previous_hash": pre_hash},
                observed={"exists": exists},
                errors=(),
            )
            output = json.dumps({"path": str(path), "status": "deleted"})
            return ToolResult("delete_file", arguments, True, output, verification=verification)
        except Exception as exc:
            return ToolResult("delete_file", arguments, False, "", str(exc))

    def _verify_file(self, path: Path, expected_bytes: bytes) -> VerificationResult:
        checks = {}
        errors = []
        observed = {}
        if not path.exists():
            checks["exists"] = False
            observed["exists"] = False
            errors.append("Missing file after write")
        else:
            checks["exists"] = True
            observed["exists"] = True
            actual = path.read_bytes()
            checks["content_matches"] = actual == expected_bytes
            observed["size"] = len(actual)
            observed["hash"] = hashlib.sha256(actual).hexdigest()
            expected_hash = hashlib.sha256(expected_bytes).hexdigest()
            checks["hash_matches"] = actual == expected_bytes
            if not checks["hash_matches"]:
                errors.append("File content does not match expected content")
        return VerificationResult(
            passed=all(checks.values()),
            checks=checks,
            expected={"size": len(expected_bytes), "hash": hashlib.sha256(expected_bytes).hexdigest()},
            observed=observed,
            errors=tuple(errors),
        )

    def _execute_code(self, arguments: dict[str, Any], context: PermissionContext, policy: PathPolicy) -> ToolResult:
        try:
            if arguments.get("timeout_seconds", 0) <= 0:
                arguments["timeout_seconds"] = 2.0
            timeout_seconds = float(arguments.get("timeout_seconds", 2.0))
            memory_limit_mb = int(arguments.get("memory_limit_mb", 128))
            cpu_time_seconds = int(arguments.get("cpu_time_seconds", 1))
            code = arguments["code"]
            temp_dir = Path(tempfile.mkdtemp(prefix="sandbox_"))
            script_path = temp_dir / "sandbox_code.py"
            script_path.write_text(code, encoding="utf-8")
            launcher = temp_dir / "launcher.py"
            launcher.write_text(
                """
import os
import resource
import sys
from pathlib import Path
code_path = Path(sys.argv[1])
cpu_limit = int(sys.argv[2])
memory_limit = int(sys.argv[3]) * 1024 * 1024
try:
    resource.setrlimit(resource.RLIMIT_CPU, (cpu_limit, cpu_limit))
    resource.setrlimit(resource.RLIMIT_AS, (memory_limit, memory_limit))
except Exception:
    pass
safe_builtins = {
    'abs': abs,
    'all': all,
    'any': any,
    'bool': bool,
    'dict': dict,
    'float': float,
    'int': int,
    'len': len,
    'list': list,
    'max': max,
    'min': min,
    'range': range,
    'str': str,
    'sum': sum,
    'print': print,
}
namespace = {'__builtins__': safe_builtins}
try:
    source = code_path.read_text(encoding='utf-8')
    exec(compile(source, str(code_path), 'exec'), namespace, namespace)
    sys.exit(0)
except Exception as exc:
    sys.stderr.write(str(exc))
    sys.exit(1)
""",
                encoding="utf-8",
            )
            env = {"PYTHONIOENCODING": "utf-8", "PATH": os.environ.get("PATH", "")}
            process = subprocess.run(
                [os.environ.get("PYTHON_EXE", subprocess.check_output(["which", "python3"]).decode().strip()), str(launcher), str(script_path), str(cpu_time_seconds), str(memory_limit_mb)],
                cwd=str(temp_dir),
                env=env,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
            )
            stdout = process.stdout
            stderr = process.stderr
            exit_code = process.returncode
            reason = "completed" if exit_code == 0 else "error"
            if process.returncode != 0:
                reason = "terminated" if process.returncode != 0 else "completed"
            output = json.dumps({"stdout": stdout[:10000], "stderr": stderr[:10000], "exit_code": exit_code, "reason": reason})
            return ToolResult("execute_code", arguments, True, output)
        except subprocess.TimeoutExpired as exc:
            return ToolResult("execute_code", arguments, False, "", f"timeout after {timeout_seconds} seconds")
        except Exception as exc:
            return ToolResult("execute_code", arguments, False, "", str(exc))

    def _api_request(self, arguments: dict[str, Any], context: PermissionContext, policy: PathPolicy) -> ToolResult:
        try:
            if httpx is None:
                raise RuntimeError("httpx is required for api_request")
            url = arguments["url"]
            parsed = httpx.URL(url)
            if parsed.host not in context.allowed_hosts:
                raise PermissionError(f"Host {parsed.host} is not allowed")
            if parsed.scheme not in {"http", "https"}:
                raise ValueError("Unsupported protocol")
            method = arguments["method"].upper()
            headers = arguments.get("headers", {})
            body = arguments.get("body", None)
            response = httpx.request(method, url, headers=headers, json=body, timeout=arguments.get("timeout_seconds", 5.0))
            output_body = None
            try:
                output_body = response.json()
            except Exception:
                output_body = response.text
            output = json.dumps({"status": str(response.status_code), "body": output_body})
            return ToolResult("api_request", arguments, response.is_success, output)
        except Exception as exc:
            return ToolResult("api_request", arguments, False, "", str(exc))

    def run(self, tool_name: str, input_data: str, permission_context: PermissionContext | None = None, policy: PathPolicy | None = None) -> ToolResult:
        try:
            tool = self.lookup(tool_name)
        except KeyError as exc:
            return ToolResult(tool_name, input_data, False, "", str(exc))
        try:
            arguments: dict[str, Any]
            if isinstance(input_data, str) and input_data.strip().startswith("{"):
                arguments = json.loads(input_data)
            else:
                arguments = {"input": input_data}
            policy = policy or PathPolicy(read_roots=[Path.cwd()], write_roots=[Path.cwd()])
            if permission_context is None:
                permission_context = PermissionContext(
                    principal_id="anonymous",
                    granted_permissions=tool.spec.required_permissions,
                    allowed_directories=(Path.cwd(),),
                    allowed_hosts=("localhost", "127.0.0.1", "api.placeholder.local"),
                    session_id="anonymous",
                    expires_at=None,
                    valid=True,
                )
            return tool.executor(arguments, permission_context, policy)
        except Exception as exc:
            return ToolResult(tool_name, input_data, False, "", traceback.format_exc())


class ToolExecutor:
    def __init__(self, registry: ToolRegistry, policy: PathPolicy, audit_log: AuditLogger) -> None:
        self.registry = registry
        self.policy = policy
        self.audit_log = audit_log

    def _check_permissions(self, spec: ToolSpec, context: PermissionContext) -> None:
        if not context.valid:
            raise PermissionError("Permission context is invalid")
        if context.expires_at is not None and time.time() > context.expires_at:
            raise PermissionError("Permission context has expired")
        missing = spec.required_permissions - context.granted_permissions
        if missing:
            raise PermissionError(f"Missing permissions: {sorted(missing)}")

    def invoke(self, tool_name: str, arguments: dict[str, Any], permission_context: PermissionContext) -> ToolResult:
        tool = self.registry.lookup(tool_name)
        normalized_args = redact_sensitive_data(arguments)
        permission_decision = "allowed"
        start_time = time.time()
        pre_action = {"allowed_directories": [str(path) for path in permission_context.allowed_directories], "allowed_hosts": list(permission_context.allowed_hosts)}
        post_action: dict[str, Any] = {}
        error: str | None = None
        result: ToolResult
        try:
            self._check_permissions(tool.spec, permission_context)
            if not self.registry._validate_input_schema(tool.spec, normalized_args):
                raise ValueError("Input schema validation failed")
            result = tool.executor(normalized_args, permission_context, self.policy)
            if result.success and isinstance(result.output, str):
                try:
                    output_obj = json.loads(result.output)
                except Exception:
                    output_obj = {"raw": result.output}
            else:
                output_obj = {}
            if not self.registry._validate_output_schema(tool.spec, output_obj):
                raise ValueError("Output schema validation failed")
            post_action = {"result_summary": result.output[:500]}
        except Exception as exc:
            result = ToolResult(tool_name, arguments, False, "", str(exc))
            permission_decision = "denied" if isinstance(exc, PermissionError) else "failed"
            error = str(exc)
        end_time = time.time()
        event_id = uuid.uuid4().hex
        record = AuditRecord(
            event_id=event_id,
            timestamp=start_time,
            session_id=permission_context.session_id,
            principal_id=permission_context.principal_id,
            tool_name=tool_name,
            effect=tool.spec.effect.value,
            risk=tool.spec.risk.value,
            arguments=arguments,
            redacted_arguments=redact_sensitive_data(arguments),
            permission_decision=permission_decision,
            start_time=start_time,
            end_time=end_time,
            result={"success": result.success, "output": result.output[:1000]},
            verification=result.verification.__dict__ if result.verification else None,
            error=error or result.error,
            pre_action=pre_action,
            post_action=post_action,
            previous_hash=self.audit_log._read_previous_hash(),
            record_hash="",
        )
        self.audit_log.append(record)
        return result


class APIConnector:
    REQUEST_SCHEMA = {
        "type": "object",
        "properties": {
            "method": {"type": "string"},
            "url": {"type": "string"},
            "headers": {"type": "object"},
            "body": {},
        },
        "required": ["method", "url"],
        "additionalProperties": False,
    }

    RESPONSE_SCHEMA = {
        "type": "object",
        "properties": {
            "status": {"type": "string"},
            "body": {},
        },
        "required": ["status", "body"],
        "additionalProperties": True,
    }

    def __init__(
        self,
        allowed_hosts: tuple[str, ...] = (),
        allowed_methods: tuple[str, ...] = ("GET", "POST"),
        allowed_schemes: tuple[str, ...] = ("http", "https"),
        base_url: str | None = None,
    ) -> None:
        self.allowed_hosts = allowed_hosts
        self.allowed_methods = allowed_methods
        self.allowed_schemes = allowed_schemes
        self.base_url = base_url
        if self.base_url is not None:
            parsed = urlparse(self.base_url)
            if parsed.hostname:
                self.allowed_hosts = tuple(sorted(set(self.allowed_hosts) | {parsed.hostname}))

    def _validate_request(self, request_obj: dict[str, Any]) -> bool:
        if jsonschema is None:
            return True
        try:
            jsonschema.validate(instance=request_obj, schema=self.REQUEST_SCHEMA)
            return True
        except Exception:
            return False

    def _validate_response(self, response_obj: dict[str, Any]) -> bool:
        if jsonschema is None:
            return True
        try:
            jsonschema.validate(instance=response_obj, schema=self.RESPONSE_SCHEMA)
            return True
        except Exception:
            return False

    def call(
        self,
        method: str,
        url: str,
        headers: dict[str, str] | None = None,
        body: Any = None,
        timeout_seconds: float = 5.0,
        permission_context: PermissionContext | None = None,
    ) -> dict[str, Any]:
        if self.base_url and not url.lower().startswith(("http://", "https://")):
            url = urljoin(self.base_url.rstrip("/") + "/", url.lstrip("/"))
        request_obj = {"method": method, "url": url, "headers": headers or {}, "body": body}
        if not self._validate_request(request_obj):
            raise ValueError("API request payload does not match the expected schema")
        parsed = urlparse(url)
        if parsed.scheme not in self.allowed_schemes:
            raise ValueError("Unsupported protocol")
        host = parsed.hostname or ""
        if self.allowed_hosts and host not in self.allowed_hosts:
            raise PermissionError(f"Host {host} is not allowed")
        if permission_context is not None and host not in permission_context.allowed_hosts:
            raise PermissionError(f"Host {host} is not explicitly allowed for this principal")
        if method.upper() not in self.allowed_methods:
            raise PermissionError(f"Method {method.upper()} is not allowed")
        if permission_context is not None and "tool.network" not in permission_context.granted_permissions:
            raise PermissionError("Network permission is required")
        if host.endswith("placeholder.local") or host in {"localhost", "127.0.0.1"}:
            return {"status": "ok", "body": {"endpoint": url, "payload": body}}
        if httpx is None:
            raise RuntimeError("httpx is required for APIConnector")
        response = httpx.request(method.upper(), url, headers=headers or {}, json=body, timeout=timeout_seconds, follow_redirects=False)
        body_content: Any
        try:
            body_content = response.json()
        except Exception:
            body_content = response.text
        result = {"status": str(response.status_code), "body": body_content}
        if not self._validate_response(result):
            raise ValueError("Response schema validation failed")
        return result


class SelfVerifier:
    TOOL_RESULT_SCHEMA = {
        "type": "object",
        "properties": {
            "tool_name": {"type": "string"},
            "success": {"type": "boolean"},
            "output": {"type": "string"},
            "error": {"anyOf": [{"type": "string"}, {"type": "null"}]},
        },
        "required": ["tool_name", "success", "output", "error"],
        "additionalProperties": False,
    }

    def __init__(self, tools: ToolRegistry, api_connector: APIConnector) -> None:
        self.tools = tools
        self.api_connector = api_connector

    def _validate_tool_result(self, result: ToolResult) -> bool:
        if jsonschema is None:
            return True
        try:
            jsonschema.validate(
                instance={
                    "tool_name": result.tool_name,
                    "success": result.success,
                    "output": result.output,
                    "error": result.error,
                },
                schema=self.TOOL_RESULT_SCHEMA,
            )
            return True
        except Exception:
            return False

    def verify_tool(self, tool_name: str, input_data: str, expected_contains: str) -> dict[str, Any]:
        result = self.tools.run(tool_name, input_data)
        schema_valid = self._validate_tool_result(result)
        verified = result.success and expected_contains in result.output and schema_valid
        return {
            "tool_name": tool_name,
            "success": result.success,
            "output": result.output,
            "error": result.error,
            "schema_valid": schema_valid,
            "expected_contains": expected_contains,
            "verified": verified,
        }

    def verify_api_connector(self, endpoint: str, payload: dict[str, Any]) -> dict[str, Any]:
        response = self.api_connector.call("POST", endpoint, headers={"Content-Type": "application/json"}, body=payload)
        response_schema_valid = self.api_connector._validate_response(response)
        valid = response.get("status") == "ok" and response_schema_valid
        return {
            "connector": self.api_connector.allowed_hosts,
            "endpoint": endpoint,
            "payload": payload,
            "response": response,
            "response_schema_valid": response_schema_valid,
            "verified": valid,
        }

    def self_check(self) -> dict[str, Any]:
        self.tools.register_builtin_tools()
        results = []
        results.append(self.verify_tool("list_directory", json.dumps({"path": "."}), "files"))
        return {
            "tool_verification": results,
            "overall_verified": all(item["verified"] for item in results),
            "schema_valid": all(item.get("schema_valid", False) for item in results),
        }
