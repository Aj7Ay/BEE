from __future__ import annotations

import json as jsonlib
import re
from pathlib import Path

from pydantic import BaseModel


class Dependency(BaseModel):
    package: str
    version: str
    source: str  # "pypi", "npm", "conda"
    direct: bool = True
    license: str | None = None
    vulnerability: str | None = None
    path: str = ""  # manifest file path where dependency was declared


def parse_requirements_txt(path: Path) -> list[Dependency]:
    """Parse requirements.txt format."""
    deps = []
    content = path.read_text(errors="replace")
    for line in content.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("-"):
            continue
        m = re.match(r"^([a-zA-Z0-9_.-]+)\s*([><=~!]+)\s*([^\s,;#]+)", line)
        if m:
            deps.append(Dependency(
                package=m.group(1),
                version=m.group(3),
                source="pypi",
                path=str(path),
            ))
    return deps


def parse_pyproject(path: Path) -> list[Dependency]:
    """Parse pyproject.toml dependencies from [project.dependencies]."""
    try:
        import tomllib
    except ImportError:
        try:
            import tomli as tomllib  # type: ignore
        except ImportError:
            return []

    deps = []
    try:
        data = tomllib.loads(path.read_text())
        dep_list = data.get("project", {}).get("dependencies", [])
        for dep_str in dep_list:
            m = re.match(r"^([a-zA-Z0-9_.-]+)\s*([><=~!][^\s]*)?", dep_str)
            if m:
                deps.append(Dependency(
                    package=m.group(1),
                    version=m.group(2) or "",
                    source="pypi",
                    path=str(path),
                ))
    except Exception:
        pass
    return deps


def parse_package_json(path: Path) -> list[Dependency]:
    """Parse package.json dependencies from dependencies + devDependencies."""
    deps = []
    try:
        data = jsonlib.loads(path.read_text())
        for section in ("dependencies", "devDependencies"):
            for pkg, version in data.get(section, {}).items():
                deps.append(Dependency(
                    package=pkg,
                    version=version,
                    source="npm",
                    path=str(path),
                ))
    except Exception:
        pass
    return deps


def parse_pipfile_lock(path: Path) -> list[Dependency]:
    """Parse Pipfile.lock format."""
    deps = []
    try:
        data = jsonlib.loads(path.read_text())
        for section in ("default", "develop"):
            for pkg, info in data.get(section, {}).items():
                deps.append(Dependency(
                    package=pkg,
                    version=info.get("version", "").lstrip("="),
                    source="pypi",
                    path=str(path),
                ))
    except Exception:
        pass
    return deps


def parse_poetry_lock(path: Path) -> list[Dependency]:
    """Parse poetry.lock format."""
    deps = []
    try:
        import tomllib
    except ImportError:
        try:
            import tomli as tomllib  # type: ignore
        except ImportError:
            return []

    try:
        data = tomllib.loads(path.read_text())
        for pkg in data.get("package", []):
            deps.append(Dependency(
                package=pkg["name"],
                version=pkg.get("version", ""),
                source="pypi",
                path=str(path),
            ))
    except Exception:
        pass
    return deps


def parse_environment_yml(path: Path) -> list[Dependency]:
    """Parse conda environment.yml dependencies."""
    deps = []
    content = path.read_text(errors="replace")
    in_deps = False
    for line in content.splitlines():
        stripped = line.strip()
        if stripped == "dependencies:":
            in_deps = True
            continue
        if in_deps:
            if not line.startswith(" ") and not line.startswith("\t") and stripped:
                break
            m = re.match(r"^\s+([a-zA-Z0-9_.-]+)\s*=?=?\s*([^\s#]+)", line)
            if m:
                deps.append(Dependency(
                    package=m.group(1),
                    version=m.group(2),
                    source="conda",
                    path=str(path),
                ))
    return deps


DEP_MANIFESTS: dict[str, callable] = {
    "requirements.txt": parse_requirements_txt,
    "pyproject.toml": parse_pyproject,
    "package.json": parse_package_json,
    "Pipfile.lock": parse_pipfile_lock,
    "poetry.lock": parse_poetry_lock,
    "environment.yml": parse_environment_yml,
}


def scan_dependencies(scan_root: Path) -> list[Dependency]:
    """Scan a directory for dependency manifest files and parse them."""
    all_deps: list[Dependency] = []

    for manifest_name, parser in DEP_MANIFESTS.items():
        manifest_path = scan_root / manifest_name
        if manifest_path.is_file():
            all_deps.extend(parser(manifest_path))

    return all_deps
