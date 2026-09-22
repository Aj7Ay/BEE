from __future__ import annotations

from pathlib import Path

import typer

from bee.cli.severity import FAIL_ON_HELP, exit_if_threshold_met, parse_severity_option
from bee.cli.state import OutputFormat
from bee.evidence.finding import Severity
from bee.scanning.config import ScanConfig
from bee.scanning.orchestrator import ScanOrchestrator
from bee.sources.huggingface import HuggingFaceSource

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

console = Console()


def huggingface_command(
    ctx: typer.Context,
    repo: str = typer.Argument(..., help="HuggingFace repo in format org/model or hf://org/model"),
    policy: Path = typer.Option(None, "--policy", help="Policy YAML file for vetting gate."),
    fail_on: str | None = typer.Option(None, "--fail-on", help=FAIL_ON_HELP),
    output: Path = typer.Option(None, "-o", "--output", help="Output directory for evidence files."),
) -> None:
    """Vet models from HuggingFace Hub.

    Examples:
    - bee hf meta-llama/Llama-2-7b
    - bee hf NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO --policy policy.yaml
    - bee hf facebook/opt-350m --fail-on high
    """
    state = ctx.obj

    # Normalize repo URL
    if not repo.startswith("hf://") and not repo.startswith("huggingface://"):
        repo = f"hf://{repo}"

    # Load policy if provided
    policy_obj = None
    if policy:
        from bee.policy.loader import load_policy
        try:
            policy_obj = load_policy(policy)
        except FileNotFoundError:
            typer.echo(f"Error: policy file not found: {policy}", err=True)
            raise typer.Exit(code=2)
        except ValueError as e:
            typer.echo(f"Error: invalid policy file: {e}", err=True)
            raise typer.Exit(code=2)
        except Exception as e:
            typer.echo(f"Error: could not load policy {policy}: {e}", err=True)
            raise typer.Exit(code=2)

    threshold = parse_severity_option(fail_on) if fail_on is not None else None

    config = ScanConfig(
        fail_on=fail_on,
        deterministic=False,
        follow_symlinks=False,
        output_dir=output,
    )

    # Initialize HuggingFace source
    try:
        source = HuggingFaceSource(repo)
    except Exception as e:
        typer.echo(f"Error: invalid HuggingFace repo: {e}", err=True)
        raise typer.Exit(code=2)

    # Scan the model
    orchestrator = ScanOrchestrator(config)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
    ) as progress:
        task = progress.add_task("Fetching model from HuggingFace Hub...", total=None)
        try:
            run_result = orchestrator.scan_source(source, policy=policy_obj)
        except Exception as e:
            typer.echo(f"Error: failed to scan model: {e}", err=True)
            raise typer.Exit(code=1)
        finally:
            source.cleanup()

    if state.output_format is OutputFormat.JSON:
        import json as jsonlib
        output_dict = run_result.model_dump(mode="json")
        if run_result.decision:
            verdict = run_result.decision
        else:
            if run_result.severity_count(Severity.CRITICAL) > 0 or run_result.severity_count(Severity.HIGH) > 0:
                verdict = "block"
            elif run_result.severity_count(Severity.MEDIUM) > 0:
                verdict = "review"
            else:
                verdict = "allow"
        output_dict["verdict"] = verdict
        typer.echo(jsonlib.dumps(output_dict, indent=2))
    else:
        from bee.reports.terminal import render_run
        render_run(run_result)

        summary_table = Table(title="HuggingFace Model Scan Summary")
        summary_table.add_column("Metric", style="cyan")
        summary_table.add_column("Value", style="green")
        summary_table.add_row("Repository", source.repo_id)
        summary_table.add_row("Total Findings", str(len(run_result.findings)))
        summary_table.add_row("Critical", str(run_result.severity_count(Severity.CRITICAL)))
        summary_table.add_row("High", str(run_result.severity_count(Severity.HIGH)))
        summary_table.add_row("Medium", str(run_result.severity_count(Severity.MEDIUM)))
        summary_table.add_row("Low", str(run_result.severity_count(Severity.LOW)))
        summary_table.add_row("Info", str(run_result.severity_count(Severity.INFO)))
        if run_result.decision:
            summary_table.add_row("Verdict", run_result.decision.upper())
        console.print(summary_table)

    if run_result.severity_count(Severity.CRITICAL) > 0 or run_result.severity_count(Severity.HIGH) > 0:
        if not policy_obj or run_result.decision != "allow":
            raise typer.Exit(code=1)

    exit_if_threshold_met(run_result.findings, threshold)
