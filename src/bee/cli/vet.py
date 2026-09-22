from __future__ import annotations

from pathlib import Path

import typer

from bee.cli.severity import FAIL_ON_HELP, exit_if_threshold_met, parse_severity_option
from bee.cli.state import OutputFormat
from bee.core.run import Run
from bee.evidence.finding import Severity
from bee.scanning.config import ScanConfig
from bee.scanning.orchestrator import ScanOrchestrator

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich.panel import Panel

console = Console()


def run(target: Path, db_path: Path | None = None) -> Run | None:
    """Scan a target and return the Run object (no CLI rendering)."""
    if db_path is None:
        db_path = Path(".bee/bee.db")
    config = ScanConfig(
        fail_on=None,
        deterministic=False,
        follow_symlinks=False,
        output_dir=None,
    )
    orchestrator = ScanOrchestrator(config)
    return orchestrator.scan_local(target, workspace_dir=db_path.parent)


def vet_command(
    ctx: typer.Context,
    path: Path = typer.Argument(..., exists=True, help="File or directory to vet."),
    db: Path = typer.Option(None, "--db", help="Path to the BEE run database."),
    fail_on: str | None = typer.Option(None, "--fail-on", help=FAIL_ON_HELP),
    deterministic: bool = typer.Option(False, "--deterministic", help="Stable run ID for diffing."),
    follow_symlinks: bool = typer.Option(False, "--follow-symlinks", help="Follow escaping symlinks."),
    output: Path = typer.Option(None, "-o", "--output", help="Output directory for evidence files."),
    policy: Path = typer.Option(None, "--policy", help="Policy YAML file for vetting gate."),
) -> None:
    """Vet an AI model artifact for security, integrity, and provenance.

    Scans local files and directories. For remote sources use:
    - Hugging Face: bee vet hf://org/model (future)
    - Ollama: bee vet ollama:qwen3:8b (future)
    """
    state = ctx.obj
    threshold = parse_severity_option(fail_on) if fail_on is not None else None
    db_path = db or state.db_path
    scan_root = path if path.is_dir() else path.parent

    # Load policy if provided
    policy_obj = None
    if policy:
        from bee.policy.loader import load_policy
        policy_obj = load_policy(policy)

    config = ScanConfig(
        fail_on=fail_on,
        deterministic=deterministic,
        follow_symlinks=follow_symlinks,
        output_dir=output,
    )

    orchestrator = ScanOrchestrator(config)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
    ) as progress:
        task = progress.add_task("Scanning target...", total=None)
        run_result = orchestrator.scan_local(path, workspace_dir=db_path.parent, policy=policy_obj)
        progress.update(task, description="Analyzing dependencies...")
        progress.update(task, description="Checking vulnerabilities...")
        if policy_obj:
            progress.update(task, description="Evaluating policy...")

    if state.output_format is OutputFormat.JSON:
        import json as jsonlib
        output_dict = run_result.model_dump(mode="json")
        output_dict["verdict"] = run_result.decision or "allow"
        typer.echo(jsonlib.dumps(output_dict, indent=2))
    else:
        from bee.reports.terminal import render_run
        render_run(run_result)

        summary_table = Table(title="Scan Summary")
        summary_table.add_column("Metric", style="cyan")
        summary_table.add_column("Value", style="green")
        summary_table.add_row("Target", str(run_result.target))
        summary_table.add_row("Total Findings", str(len(run_result.findings)))
        summary_table.add_row("Critical", str(run_result.severity_count(Severity.CRITICAL)))
        summary_table.add_row("High", str(run_result.severity_count(Severity.HIGH)))
        summary_table.add_row("Medium", str(run_result.severity_count(Severity.MEDIUM)))
        summary_table.add_row("Low", str(run_result.severity_count(Severity.LOW)))
        summary_table.add_row("Info", str(run_result.severity_count(Severity.INFO)))
        if run_result.decision:
            summary_table.add_row("Verdict", run_result.decision.upper())
        console.print(summary_table)

    # Fail-closed: exit non-zero if any critical/high findings (unless explicitly allowed by policy)
    if run_result.severity_count(Severity.CRITICAL) > 0 or run_result.severity_count(Severity.HIGH) > 0:
        if not policy_obj or run_result.decision != "allow":
            raise typer.Exit(code=1)

    exit_if_threshold_met(run_result.findings, threshold)
