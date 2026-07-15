from __future__ import annotations

import json
import math
import traceback
from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class ToolResult:
    tool_name: str
    input_data: str
    success: bool
    output: str
    error: str | None = None


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    execute: Callable[[str], ToolResult]


class ToolRegistry:
    def __init__(self) -> None:
        self.tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self.tools[tool.name] = tool

    def list_tools(self) -> list[str]:
        return sorted(self.tools.keys())

    def run(self, tool_name: str, input_data: str) -> ToolResult:
        if tool_name not in self.tools:
            return ToolResult(tool_name, input_data, False, "", f"Tool '{tool_name}' is not registered")
        try:
            return self.tools[tool_name].execute(input_data)
        except Exception as exc:
            return ToolResult(tool_name, input_data, False, "", traceback.format_exc())

    def register_builtin_tools(self) -> None:
        self.register(Tool("echo", "Return the provided input string.", self._echo))
        self.register(Tool("sum", "Sum comma-separated numeric values.", self._sum))
        self.register(Tool("python", "Execute a Python snippet and return the result.", self._execute_python))
        self.register(Tool("api_call", "Call a configured external API connector.", self._call_api))

    def _echo(self, input_data: str) -> ToolResult:
        return ToolResult("echo", input_data, True, input_data)

    def _sum(self, input_data: str) -> ToolResult:
        try:
            numbers = [float(value.strip()) for value in input_data.split(",") if value.strip()]
            total = sum(numbers)
            return ToolResult("sum", input_data, True, str(total))
        except Exception as exc:
            return ToolResult("sum", input_data, False, "", str(exc))

    def _execute_python(self, input_data: str) -> ToolResult:
        try:
            safe_globals = {
                "__builtins__": {
                    "abs": abs,
                    "min": min,
                    "max": max,
                    "sum": sum,
                    "len": len,
                    "range": range,
                    "math": math,
                }
            }
            safe_locals: dict[str, Any] = {}
            try:
                compiled = compile(input_data, "<python_tool>", "eval")
                value = eval(compiled, safe_globals, safe_locals)
            except SyntaxError:
                exec(compile(input_data, "<python_tool>", "exec"), safe_globals, safe_locals)
                value = safe_locals.get("result", "<no result variable>")
            return ToolResult("python", input_data, True, repr(value))
        except Exception as exc:
            return ToolResult("python", input_data, False, "", traceback.format_exc())

    def _call_api(self, input_data: str) -> ToolResult:
        try:
            payload = json.loads(input_data)
            endpoint = payload.get("endpoint", "unknown")
            body = payload.get("payload", {})
            response = {
                "endpoint": endpoint,
                "status": "ok",
                "payload": body,
                "message": "Simulated external API response",
            }
            return ToolResult("api_call", input_data, True, json.dumps(response))
        except Exception as exc:
            return ToolResult("api_call", input_data, False, "", traceback.format_exc())


class APIConnector:
    def __init__(self, name: str, base_url: str = "https://api.placeholder.local") -> None:
        self.name = name
        self.base_url = base_url

    def call(self, endpoint: str, payload: dict[str, Any]) -> dict[str, Any]:
        return {
            "connector": self.name,
            "endpoint": endpoint,
            "base_url": self.base_url,
            "payload": payload,
            "status": "ok",
            "summary": "External API connector stub response",
        }


class SelfVerifier:
    def __init__(self, tools: ToolRegistry, api_connector: APIConnector) -> None:
        self.tools = tools
        self.api_connector = api_connector

    def verify_tool(self, tool_name: str, input_data: str, expected_contains: str) -> dict[str, Any]:
        result = self.tools.run(tool_name, input_data)
        return {
            "tool_name": tool_name,
            "success": result.success,
            "output": result.output,
            "error": result.error,
            "expected_contains": expected_contains,
            "verified": result.success and expected_contains in result.output,
        }

    def verify_api_connector(self, endpoint: str, payload: dict[str, Any]) -> dict[str, Any]:
        response = self.api_connector.call(endpoint, payload)
        valid = response.get("status") == "ok"
        return {
            "connector": self.api_connector.name,
            "endpoint": endpoint,
            "payload": payload,
            "response": response,
            "verified": valid,
        }

    def self_check(self) -> dict[str, Any]:
        self.tools.register_builtin_tools()
        results = []
        results.append(self.verify_tool("echo", "hello world", "hello"))
        results.append(self.verify_tool("sum", "1, 2, 3", "6"))
        results.append(self.verify_tool("python", "result = 1 + 2", "3"))
        api_result = self.verify_api_connector("status", {"health": True})
        return {
            "tool_verification": results,
            "api_verification": api_result,
            "overall_verified": all(item["verified"] for item in results) and api_result["verified"],
        }
