"""
CoMLRL-style helper utilities for collaborative code generation evaluation.
"""

import ast
import re
from typing import Tuple


def cleanup_code(code: str) -> str:
    if not code:
        return ""
    # Strip markdown fences.
    code = re.sub(r"```python\s*\n?", "", code)
    code = re.sub(r"```\s*\n?", "", code)
    return code.strip()


def extract_specific_function(code: str, function_name: str) -> str:
    cleaned = cleanup_code(code)
    lines = cleaned.split("\n")
    function_lines = []
    in_target = False
    base_indent = 0
    for line in lines:
        if re.match(rf"^(\s*)def\s+{re.escape(function_name)}\s*\(", line):
            function_lines = [line]
            in_target = True
            base_indent = len(line) - len(line.lstrip())
            continue
        if in_target:
            if (
                line.strip() == ""
                or line.startswith(" " * (base_indent + 1))
                or line.startswith("\t")
            ):
                function_lines.append(line)
            else:
                break
    return "\n".join(function_lines).strip()


def check_function_definition(code: str, function_name: str) -> Tuple[bool, str]:
    func_code = extract_specific_function(code, function_name)
    if not func_code:
        return False, f"{function_name} not defined"
    if "return" not in func_code:
        return False, f"{function_name} missing return"
    return True, f"{function_name} defined"


def check_syntax(code: str) -> Tuple[bool, str]:
    try:
        ast.parse(code)
        return True, "syntax ok"
    except SyntaxError as exc:  # noqa: BLE001
        return False, f"syntax error: {exc}"


def combine_code(aux_code: str, main_code: str) -> str:
    aux_clean = cleanup_code(aux_code)
    main_clean = cleanup_code(main_code)
    parts = []
    if aux_clean:
        parts.append(aux_clean)
    if main_clean:
        parts.append(main_clean)
    return "\n\n".join(parts)


def check_aux_function_usage(main_code: str, aux_name: str = "aux") -> bool:
    if not main_code:
        return False
    pattern = rf"\b{re.escape(aux_name)}\s*\("
    return re.search(pattern, main_code) is not None


def is_wrapper_function(main_code: str, aux_name: str = "aux") -> bool:
    if not main_code:
        return True
    lines = [line.strip() for line in main_code.split("\n") if line.strip()]
    # Remove def line
    lines = [line for line in lines if not line.startswith("def ")]
    if not lines:
        return True
    if len(lines) == 1:
        line = lines[0]
        if line.startswith("return ") and aux_name in line:
            return True
    if len(lines) == 2:
        first, second = lines
        match = re.match(rf"(\w+)\s*=\s*{re.escape(aux_name)}\s*\(", first)
        if match and second == f"return {match.group(1)}":
            return True
    return False


def check_aux_call_without_assignment(main_code: str, aux_name: str = "aux") -> bool:
    if not main_code:
        return False
    pattern = rf"^\s*{re.escape(aux_name)}\s*\([^)]*\)\s*$"
    for line in main_code.split("\n"):
        if re.match(pattern, line.strip()):
            return True
    return False
