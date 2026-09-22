from __future__ import annotations

import json as jsonlib
from pathlib import Path
from datetime import datetime, timezone

from bee.core.artifact import Artifact
from bee.core.run import Run, Provenance
from bee.evidence.custom_code import scan_code_directory
from bee.evidence.dependency import Dependency, scan_dependencies
from bee.evidence.finding import Finding, Severity, Evidence, Confidence
from bee.evidence.gguf_bounds import check_gguf_bounds
from bee.evidence.mismatch import check_mismatch
from bee.evidence.pickle_calls import check_pickle_calls
from bee.evidence.provenance import build_provenance
from bee.evidence.safetensors_bounds import check_safetensors_bounds, check_safetensors_gap
from bee.evidence.symlink import build_symlink_escape_finding, is_escaping_symlink
from bee.evidence.vulnerability import enrich_dependencies_with_vulns
from bee.scanning.config import ScanConfig
from bee.policy.rules import Policy
from bee.policy.evaluator import PolicyEvaluator, Decision, PolicyViolation
from bee.sources.base import Source


def _iter_files(target: Path, workspace_dir: Path) -> list[Path]:
    """Same logic as cli/scan.py — list all files in target, excluding .bee/ workspace."""
    workspace_abs = workspace_dir.absolute()
    if target.is_file() or target.is_symlink():
        return [target]
    results = []
    for p in target.rglob("*"):
        if not (p.is_file() or p.is_symlink()):
            continue
        if p.absolute().is_relative_to(workspace_abs):
            continue
        results.append(p)
    return sorted(results)


class ScanOrchestrator:
    """Coordinates all scanning engines for a single target."""

    def __init__(self, config: ScanConfig):
        self.config = config

    def scan_code_files(self, target: Path) -> list[Finding]:
        """Scan target directory for dangerous code patterns via custom_code module."""
        if not target.is_dir():
            return []
        return scan_code_directory(target)

    def scan_dependencies(self, target: Path) -> tuple[list[Dependency], list[dict]]:
        """Scan target for dependency manifests and enrich with vulnerability data."""
        if not target.is_dir():
            return [], []
        deps = scan_dependencies(target)
        return enrich_dependencies_with_vulns(deps)

    def scan_local(self, target: Path, workspace_dir: Path, policy: Policy | None = None) -> Run:
        """Run the full scan pipeline on a local path."""
        files = _iter_files(target, workspace_dir)
        artifacts: list[Artifact] = []
        findings: list[Finding] = []

        for file_path in files:
            # Symlink check (same as scan.py)
            scan_root = target if target.is_dir() else target
            escaping_target = is_escaping_symlink(file_path, scan_root)
            if escaping_target is not None:
                findings.append(
                    build_symlink_escape_finding(
                        str(file_path), escaping_target, content_read=self.config.follow_symlinks
                    )
                )
                if not self.config.follow_symlinks:
                    artifacts.append(Artifact.unresolved_symlink(file_path, escaping_target))
                    continue

            try:
                artifact, content = Artifact.from_file_with_content(file_path)
            except OSError as exc:
                findings.append(Finding(
                    id="BEE-IO-001",
                    severity=Severity.HIGH,
                    title="Artifact could not be read",
                    description=f"Reading this file failed: {exc}",
                    artifact_path=str(file_path),
                    evidence=[
                        Evidence(type="os_error", value=str(exc), source="local_filesystem", confidence=Confidence.VERIFIED),
                    ],
                ))
                continue

            artifacts.append(artifact)

            # Run existing scanners (EXACTLY like scan.py)
            finding = check_mismatch(artifact)
            if finding is not None:
                findings.append(finding)

            pickle_finding = check_pickle_calls(artifact, content)
            if pickle_finding is not None:
                findings.append(pickle_finding)

            bounds_finding = check_safetensors_bounds(artifact, content)
            if bounds_finding is not None:
                findings.append(bounds_finding)

            gap_finding = check_safetensors_gap(artifact, content)
            if gap_finding is not None:
                findings.append(gap_finding)

            gguf_finding = check_gguf_bounds(artifact, content)
            if gguf_finding is not None:
                findings.append(gguf_finding)

        # Build provenance (best-effort from local scan)
        provenance = self._build_local_provenance(target, artifacts)

        # Run code scanning via self.scan_code_files
        code_findings = self.scan_code_files(target)
        findings.extend(code_findings)

        # Run dependency scanning via self.scan_dependencies
        enriched_deps, vuln_findings = self.scan_dependencies(target)

        # Build mapping from package name to manifest path
        pkg_to_path = {d.package: d.path for d in enriched_deps}

        # Convert vulnerability findings to Finding objects and add to findings
        for vuln in vuln_findings:
            artifact_path = pkg_to_path.get(vuln.get("package", ""), "")
            vuln_finding = Finding(
                id=f"BEE-VULN-{vuln.get('osv_id', 'UNK')[:8]}",
                severity=self._vuln_severity_to_bee(vuln.get("severity", "unknown")),
                title=f"Vulnerability in {vuln.get('package', '?')}",
                description=vuln.get("description", ""),
                artifact_path=artifact_path,
                evidence=[
                    Evidence(
                        type="vulnerability",
                        value=vuln.get("cve_id") or vuln.get("osv_id", ""),
                        source="osv",
                        confidence=Confidence.VERIFIED,
                    ),
                ],
            )
            findings.append(vuln_finding)

        # License and model card scanning
        license_info = self._scan_license(target)
        model_card = self._scan_model_card(target)

        # Run policy evaluation against policy if provided
        decision_str: str | None = None
        policy_violations_list: list[dict] = []
        if policy is not None:
            evaluator = PolicyEvaluator()
            decision_obj, policy_violations = evaluator.evaluate(
                policy=policy,
                findings=findings,
                provenance=provenance,
                license_info=license_info,
                vulnerabilities=vuln_findings,
            )
            decision_str = decision_obj.value if decision_obj else None
            policy_violations_list = [v.model_dump(mode="json") if hasattr(v, "model_dump") else v for v in policy_violations]

        run = Run.from_scan(
            target_path=str(target),
            artifacts=artifacts,
            findings=findings,
            deterministic=self.config.deterministic,
            provenance=provenance,
            vulnerabilities=vuln_findings,
            decision=decision_str,
            policy_violations=policy_violations_list,
            license_info=license_info,
            model_card=model_card,
            dependencies=[d.model_dump(mode="json") for d in enriched_deps],
        )

        # Write evidence files if configured
        if self.config.output_dir and self.config.write_evidence_files:
            self._write_evidence_files(run, self.config.output_dir)

        return run

    def scan_source(self, source: Source, policy: Policy | None = None) -> Run:
        """Scan any Source abstraction."""
        artifacts = source.get_artifacts()
        source_info = source.get_source_info()
        model_card = source.get_model_card()
        findings: list[Finding] = []
        artifact_objs: list[Artifact] = []
        scan_root = Path.cwd()

        for artifact_path in artifacts:
            escaping_target = is_escaping_symlink(artifact_path, scan_root)
            if escaping_target is not None and not self.config.follow_symlinks:
                findings.append(
                    build_symlink_escape_finding(
                        str(artifact_path), escaping_target, content_read=self.config.follow_symlinks
                    )
                )
                artifact_objs.append(Artifact.unresolved_symlink(artifact_path, escaping_target))
                continue

            try:
                artifact, content = Artifact.from_file_with_content(artifact_path)
            except OSError as exc:
                findings.append(Finding(
                    id="BEE-IO-001",
                    severity=Severity.HIGH,
                    title="Artifact could not be read",
                    description=f"Reading this file failed: {exc}",
                    artifact_path=str(artifact_path),
                    evidence=[
                        Evidence(type="os_error", value=str(exc), source="local_filesystem", confidence=Confidence.VERIFIED),
                    ],
                ))
                continue

            artifact_objs.append(artifact)

            finding = check_mismatch(artifact)
            if finding is not None:
                findings.append(finding)

            pickle_finding = check_pickle_calls(artifact, content)
            if pickle_finding is not None:
                findings.append(pickle_finding)

            bounds_finding = check_safetensors_bounds(artifact, content)
            if bounds_finding is not None:
                findings.append(bounds_finding)

            gap_finding = check_safetensors_gap(artifact, content)
            if gap_finding is not None:
                findings.append(gap_finding)

            gguf_finding = check_gguf_bounds(artifact, content)
            if gguf_finding is not None:
                findings.append(gguf_finding)

        # Build provenance from source info
        first_artifact = artifact_objs[0] if artifact_objs else None
        if first_artifact:
            provenance = build_provenance(
                provider=source_info["provider"],
                repository=source_info.get("repository"),
                revision=source_info.get("revision"),
                artifact_path=str(first_artifact.path),
                sha256=first_artifact.sha256,
                artifact_size=first_artifact.size,
                artifact_filename=Path(first_artifact.path).name,
                acquisition_method=source.source_type,
            )
        else:
            provenance = build_provenance(
                provider=source_info["provider"],
                acquisition_method=source.source_type,
            )

        run = Run.from_scan(
            target_path=source.url if hasattr(source, "url") and source.url else source.source_type,
            artifacts=artifact_objs,
            findings=findings,
            deterministic=self.config.deterministic,
            provenance=provenance,
            vulnerabilities=[],
            model_card=model_card,
        )

        if policy is not None:
            evaluator = PolicyEvaluator()
            _, policy_violations = evaluator.evaluate(
                policy=policy,
                findings=findings,
                provenance=provenance,
                license_info=run.license_info,
                vulnerabilities=[],
            )
            if policy_violations:
                run.decision = Decision.BLOCK if any(v.action == "block" for v in policy_violations) else Decision.REVIEW if any(v.action == "review" for v in policy_violations) else Decision.ALLOW

        if self.config.output_dir and self.config.write_evidence_files:
            self._write_evidence_files(run, self.config.output_dir)

        source.cleanup()
        return run


    def _scan_license(self, target: Path) -> dict | None:
        """Scan for license info."""
        from bee.evidence.license import detect_license_in_directory, assess_license_compatibility
        licenses = detect_license_in_directory(target)
        if licenses:
            return {"detected": licenses, "compatibility": assess_license_compatibility(licenses)}
        return None

    def _scan_model_card(self, target: Path) -> dict | None:
        """Try to extract model card from target directory."""
        readme = target / "README.md"
        if readme.is_file():
            from bee.evidence.model_card import parse_model_card
            return parse_model_card(readme)
        return None

    def _vuln_severity_to_bee(self, severity: str) -> Severity:
        """Map OSV/CVE severity to BEE Severity."""
        severity_lower = str(severity).lower()
        if severity_lower == "critical":
            return Severity.CRITICAL
        elif severity_lower == "high":
            return Severity.HIGH
        elif severity_lower == "medium":
            return Severity.MEDIUM
        elif severity_lower == "low":
            return Severity.LOW
        else:
            return Severity.INFO

    def _build_local_provenance(self, target: Path, artifacts: list[Artifact]) -> Provenance:
        """Build best-effort provenance for a local scan."""
        if artifacts:
            first = artifacts[0]
            return build_provenance(
                provider="local",
                artifact_path=str(first.path),
                sha256=first.sha256,
                artifact_size=first.size,
                artifact_filename=Path(first.path).name,
                acquisition_method="local_copy",
            )
        return build_provenance(
            provider="local",
            acquisition_method="local_copy",
        )

    def _write_evidence_files(self, run: Run, output_dir: Path):
        """Write individual evidence files + bee-evidence.json."""
        from bee.evidence.provenance import build_provenance_graph

        output_dir.mkdir(parents=True, exist_ok=True)

        def _write(name: str, data):
            path = output_dir / name
            path.write_text(jsonlib.dumps(data, indent=2, default=str) + "\n")

        # Individual evidence files
        _write("artifact.json", [a.model_dump(mode="json") for a in run.artifacts])
        _write("provenance.json", run.provenance.model_dump(mode="json") if run.provenance else None)
        _write("findings.json", [f.model_dump(mode="json") for f in run.findings])
        _write("vulnerabilities.json", run.vulnerabilities)
        _write("license.json", run.license_info)
        _write("findings_by_severity.json", run.summary.findings_by_severity)

        # Combined bee-evidence.json
        evidence = {
            "bee_version": run.scanner_version,
            "created_at": run.created_at.isoformat(),
            "run_id": run.id,
            "target_path": run.target_path,
            "evidence_sha256": run.evidence_sha256,
            "artifacts": [a.model_dump(mode="json") for a in run.artifacts],
            "findings": [f.model_dump(mode="json") for f in run.findings],
            "summary": run.summary.model_dump(mode="json"),
            "provenance": run.provenance.model_dump(mode="json") if run.provenance else None,
            "decision": run.decision,
            "vulnerabilities": run.vulnerabilities,
            "license_info": run.license_info,
            "model_card": run.model_card,
            "provenance_graph": build_provenance_graph(run.provenance, has_signature=bool(run.signature)) if run.provenance else [],
            "signature": run.signature if run.signature else None,
            "public_key": run.public_key if run.public_key else None,
        }
        _write("bee-evidence.json", evidence)


def write_json(path: Path, data):
    path.write_text(jsonlib.dumps(data, indent=2, default=str) + "\n")
