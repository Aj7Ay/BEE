from __future__ import annotations

import re
from pathlib import Path

from bee.evidence.finding import Confidence, Evidence, Finding, Severity

# Each rule: (id, category, severity, regex_pattern, description, recommendation)
CODE_RULES: list[tuple[str, str, Severity, str, str, str]] = [
    (
        "BEE-CODE-001",
        "command_execution",
        Severity.CRITICAL,
        r"os\.system\s*\(",
        "Direct command execution via os.system()",
        "Review for privilege escalation or arbitrary command execution risk.",
    ),
    (
        "BEE-CODE-001",
        "command_execution",
        Severity.CRITICAL,
        r"os\.popen\s*\(",
        "Command execution via os.popen()",
        "Review for command injection risk.",
    ),
    (
        "BEE-CODE-001",
        "command_execution",
        Severity.CRITICAL,
        r"subprocess\.(run|call|check_output|check_call|Popen)\s*\(",
        "Subprocess execution detected",
        "Review subprocess arguments for injection risk.",
    ),
    (
        "BEE-CODE-002",
        "dynamic_execution",
        Severity.HIGH,
        r"\beval\s*\(",
        "Dynamic code evaluation via eval()",
        "Ensure input is sanitized before dynamic evaluation.",
    ),
    (
        "BEE-CODE-002",
        "dynamic_execution",
        Severity.HIGH,
        r"\bcompile\s*\(",
        "Dynamic code compilation via compile()",
        "Ensure compiled code comes from trusted sources.",
    ),
    (
        "BEE-CODE-002",
        "dynamic_execution",
        Severity.HIGH,
        r"\bexec\s*\(",
        "Code execution via exec()",
        "Review for arbitrary code execution risk.",
    ),
    (
        "BEE-CODE-003",
        "network_access",
        Severity.MEDIUM,
        r"requests\.(get|post|put|delete|patch|head|options)\s*\(",
        "HTTP request via requests library",
        "Verify network access is expected and uses TLS.",
    ),
    (
        "BEE-CODE-003",
        "network_access",
        Severity.MEDIUM,
        r"urllib\.(request|request\.urlopen)\s*\(",
        "HTTP request via urllib",
        "Verify network access is expected and uses TLS.",
    ),
    (
        "BEE-CODE-003",
        "network_access",
        Severity.MEDIUM,
        r"socket\.socket\s*\(",
        "Raw socket creation detected",
        "Review for unexpected network connections.",
    ),
    (
        "BEE-CODE-004",
        "credential_access",
        Severity.HIGH,
        r"os\.environ\s*\[",
        "Direct environment variable access via os.environ[]",
        "Verify this code does not leak credentials.",
    ),
    (
        "BEE-CODE-004",
        "credential_access",
        Severity.HIGH,
        r"os\.getenv\s*\(",
        "Environment variable access via os.getenv()",
        "Verify this code does not leak credentials.",
    ),
    (
        "BEE-CODE-004",
        "credential_access",
        Severity.HIGH,
        r"keyring\.",
        "Credential storage access via keyring",
        "Review credential handling for security.",
    ),
    (
        "BEE-CODE-005",
        "dynamic_imports",
        Severity.MEDIUM,
        r"__import__\s*\(",
        "Dynamic import via __import__()",
        "Ensure dynamically imported modules are from trusted sources.",
    ),
    (
        "BEE-CODE-005",
        "dynamic_imports",
        Severity.MEDIUM,
        r"importlib\.import_module\s*\(",
        "Dynamic import via importlib",
        "Ensure dynamically imported modules are from trusted sources.",
    ),
    (
        "BEE-CODE-006",
        "filesystem_modification",
        Severity.MEDIUM,
        r"os\.remove\s*\(",
        "File deletion via os.remove()",
        "Verify filesystem modifications are intentional and safe.",
    ),
    (
        "BEE-CODE-006",
        "filesystem_modification",
        Severity.MEDIUM,
        r"os\.rmdir\s*\(",
        "Directory removal via os.rmdir()",
        "Verify filesystem modifications are intentional and safe.",
    ),
    (
        "BEE-CODE-006",
        "filesystem_modification",
        Severity.MEDIUM,
        r"shutil\.(rmtree|move|copy|copytree)\s*\(",
        "Filesystem operations via shutil",
        "Verify filesystem modifications are intentional and safe.",
    ),
    (
        "BEE-CODE-007",
        "download_execute",
        Severity.CRITICAL,
        r"urllib\.request\.urlretrieve",
        "Download to file via urlretrieve",
        "Verify download source and integrity before execution.",
    ),
    (
        "BEE-CODE-007",
        "download_execute",
        Severity.CRITICAL,
        r"requests\.(get|stream)\s*\(.*\.(content|text)",
        "Download content via requests",
        "Verify download source and integrity before execution.",
    ),
    (
        "BEE-CODE-007",
        "download_execute",
        Severity.CRITICAL,
        r"\bwget\s",
        "wget command used",
        "Review for download-and-execute behavior.",
    ),
]

# File extensions to scan
SCAN_EXTENSIONS = frozenset({".py", ".ipynb", ".sh", ".ps1", ".js", ".ts"})


def scan_code_directory(root: Path) -> list[Finding]:
    """Scan all supported files in a directory tree for dangerous code patterns."""
    findings: list[Finding] = []

    for file_path in root.rglob("*"):
        if file_path.suffix.lower() not in SCAN_EXTENSIONS:
            continue
        if file_path.is_symlink():
            continue

        try:
            lines = file_path.read_text(errors="replace").splitlines()
        except OSError:
            continue

        for line_num, line in enumerate(lines, start=1):
            for rule_id, category, severity, pattern, description, recommendation in CODE_RULES:
                if re.search(pattern, line):
                    findings.append(Finding(
                        id=rule_id,
                        severity=severity,
                        title=description,
                        description=(
                            f"{description}\n"
                            f"File: {file_path}\n"
                            f"Line: {line_num}\n"
                            f"Pattern: {line.strip()}"
                        ),
                        artifact_path=str(file_path),
                        category=category,
                        line_number=line_num,
                        code_pattern=pattern,
                        recommendation=recommendation,
                        evidence=[
                            Evidence(
                                type="code_pattern",
                                value=pattern,
                                source="code_analysis",
                                confidence=Confidence.VERIFIED,
                            ),
                        ],
                    ))

    return findings
