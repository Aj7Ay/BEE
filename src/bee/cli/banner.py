"""CLI banner display."""

from bee import __version__


def show_banner() -> None:
    """Display BEE banner with version."""
    banner = f"""
[bold cyan]
    🐝  BEE v{__version__}
[/bold cyan]
[dim]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/dim]
[green]AI/ML Model Supply-Chain Security[/green]
"""
    from rich.console import Console
    console = Console()
    console.print(banner)
