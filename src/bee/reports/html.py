from __future__ import annotations

import html
from datetime import datetime
from pathlib import Path
from typing import Any

from bee.core.run import Run
from bee.evidence.finding import Severity


class HTMLReport:
    """Generate a visual HTML report from a BEE scan run."""

    SEVERITY_COLORS = {
        Severity.CRITICAL: "#dc2626",
        Severity.HIGH: "#ea580c",
        Severity.MEDIUM: "#d97706",
        Severity.LOW: "#059669",
        Severity.INFO: "#6366f1",
    }

    DECISION_STYLES = {
        "allow": ("&#10004; Allow", "#059669"),
        "review": ("&#9888; Review", "#d97706"),
        "block": ("&#10060; Block", "#dc2626"),
    }

    def __init__(self, run: Run):
        self.run = run

    def generate(self, output_path: Path | None = None) -> str:
        """Generate HTML report string and optionally save to file."""
        html_str = self._build_html()

        if output_path:
            output_path.write_text(html_str, encoding="utf-8")

        return html_str

    def _build_html(self) -> str:
        r = self.run
        decision_text, decision_color = self.DECISION_STYLES.get(
            r.decision, ("Unknown", "#6b7280")
        )

        # Build findings rows
        findings_rows = self._build_findings_rows()

        # Build provenance section
        provenance_html = self._build_provenance()

        # Build dependencies section
        deps_html = self._build_dependencies()

        # Build vulnerabilities section
        vulns_html = self._build_vulnerabilities()

        timestamp = datetime.fromtimestamp(r.timestamp).strftime("%Y-%m-%d %H:%M:%S")

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>BEE Security Report - {html.escape(r.target.name if hasattr(r.target, 'name') else str(r.target))}</title>
<style>
    * {{ margin: 0; padding: 0; box-sizing: border-box; }}
    body {{
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        background: #0f172a;
        color: #e2e8f0;
        line-height: 1.6;
        padding: 2rem;
    }}
    .container {{ max-width: 1200px; margin: 0 auto; }}
    h1 {{
        font-size: 2rem;
        margin-bottom: 0.5rem;
        color: #38bdf8;
    }}
    .subtitle {{ color: #94a3b8; margin-bottom: 2rem; }}
    .decision-banner {{
        padding: 1.5rem;
        border-radius: 12px;
        margin-bottom: 2rem;
        font-size: 1.25rem;
        font-weight: 600;
        color: white;
        background: {decision_color};
    }}
    .summary-grid {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
        gap: 1rem;
        margin-bottom: 2rem;
    }}
    .summary-card {{
        background: #1e293b;
        border-radius: 10px;
        padding: 1.25rem;
        text-align: center;
    }}
    .summary-card .count {{
        font-size: 2.5rem;
        font-weight: 700;
        color: {self.SEVERITY_COLORS.get(Severity.CRITICAL, '#dc2626')};
    }}
    .summary-card.high .count {{ color: {self.SEVERITY_COLORS[Severity.HIGH]} }}
    .summary-card.medium .count {{ color: {self.SEVERITY_COLORS[Severity.MEDIUM]} }}
    .summary-card.low .count {{ color: {self.SEVERITY_COLORS[Severity.LOW]} }}
    .summary-card.info .count {{ color: {self.SEVERITY_COLORS[Severity.INFO]} }}
    .summary-card .label {{ color: #94a3b8; font-size: 0.875rem; margin-top: 0.25rem; }}
    table {{
        width: 100%;
        border-collapse: collapse;
        background: #1e293b;
        border-radius: 10px;
        overflow: hidden;
        margin-bottom: 2rem;
    }}
    th {{
        background: #334155;
        padding: 0.75rem 1rem;
        text-align: left;
        font-weight: 600;
        color: #cbd5e1;
        font-size: 0.875rem;
    }}
    td {{
        padding: 0.75rem 1rem;
        border-top: 1px solid #334155;
        font-size: 0.875rem;
    }}
    .severity-badge {{
        display: inline-block;
        padding: 0.2rem 0.6rem;
        border-radius: 6px;
        font-size: 0.75rem;
        font-weight: 600;
        color: white;
        text-transform: uppercase;
    }}
    .section {{
        background: #1e293b;
        border-radius: 10px;
        padding: 1.5rem;
        margin-bottom: 2rem;
    }}
    .section h2 {{
        color: #38bdf8;
        margin-bottom: 1rem;
        font-size: 1.25rem;
    }}
    .metadata {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem; }}
    .meta-item {{ background: #0f172a; padding: 0.75rem 1rem; border-radius: 8px; }}
    .meta-item .key {{ color: #94a3b8; font-size: 0.75rem; }}
    .meta-item .value {{ color: #e2e8f0; font-weight: 500; }}
    .vuln-row {{ background: #1e293b; border-radius: 8px; padding: 1rem; margin-bottom: 0.5rem; border-left: 4px solid {self.SEVERITY_COLORS[Severity.HIGH]}; }}
    .vuln-row.medium {{ border-left-color: {self.SEVERITY_COLORS[Severity.MEDIUM]} }}
    footer {{ text-align: center; color: #475569; margin-top: 3rem; font-size: 0.875rem; }}
    .collapsible {{ cursor: pointer; user-select: none; }}
    .collapsible:hover {{ color: #38bdf8; }}
</style>
</head>
<body>
<div class="container">
    <h1>BEE Security Report</h1>
    <p class="subtitle">Target: {html.escape(str(r.target))} | Scanned: {timestamp}</p>

    <div class="decision-banner">{decision_text}</div>

    <div class="summary-grid">
        <div class="summary-card">
            <div class="count">{len(r.findings)}</div>
            <div class="label">Total Findings</div>
        </div>
        <div class="summary-card">
            <div class="count">{r.severity_count(Severity.CRITICAL)}</div>
            <div class="label">Critical</div>
        </div>
        <div class="summary-card high">
            <div class="count">{r.severity_count(Severity.HIGH)}</div>
            <div class="label">High</div>
        </div>
        <div class="summary-card medium">
            <div class="count">{r.severity_count(Severity.MEDIUM)}</div>
            <div class="label">Medium</div>
        </div>
        <div class="summary-card low">
            <div class="count">{r.severity_count(Severity.LOW)}</div>
            <div class="label">Low</div>
        </div>
        <div class="summary-card info">
            <div class="count">{r.severity_count(Severity.INFO)}</div>
            <div class="label">Info</div>
        </div>
    </div>

    <div class="section">
        <h2>Findings</h2>
        <table>
            <thead>
                <tr>
                    <th>ID</th>
                    <th>Severity</th>
                    <th>Category</th>
                    <th>Message</th>
                </tr>
            </thead>
            <tbody>
{findings_rows}
            </tbody>
        </table>
    </div>

{provenance_html}
{deps_html}
{vulns_html}

    <footer>BEE Security Scanner v{r.version} | {timestamp}</footer>
</div>
</body>
</html>"""

    def _build_findings_rows(self) -> str:
        rows = []
        for f in sorted(self.run.findings, key=lambda x: x.severity.value):
            color = self.SEVERITY_COLORS.get(f.severity, "#6b7280")
            rows.append(f"""                <tr>
                    <td><code>{html.escape(f.id)}</code></td>
                    <td><span class="severity-badge" style="background:{color}">{html.escape(f.severity.value)}</span></td>
                    <td>{html.escape(f.category)}</td>
                    <td>{html.escape(f.title)}</td>
                </tr>""")
        if not rows:
            rows = "                <tr><td colspan=4 style='text-align:center;color:#94a3b8'>No findings</td></tr>"
        return "\n".join(rows)

    def _build_provenance(self) -> str:
        if not self.run.provenance:
            return ""
        p = self.run.provenance
        source = p.source
        return f"""    <div class="section">
        <h2>Provenance</h2>
        <div class="metadata">
            <div class="meta-item"><div class="key">Provider</div><div class="value">{html.escape(str(source.provider))}</div></div>
            <div class="meta-item"><div class="key">Repository</div><div class="value">{html.escape(str(source.repository or 'N/A'))}</div></div>
            <div class="meta-item"><div class="key">Revision</div><div class="value">{html.escape(str(source.revision or 'N/A'))}</div></div>
            <div class="meta-item"><div class="key">SHA-256</div><div class="value"><code>{html.escape(p.artifact.sha256[:32] + '...')}</code></div></div>
            <div class="meta-item"><div class="key">Size</div><div class="value">{p.artifact.size:,} bytes</div></div>
        </div>
    </div>"""

    def _build_dependencies(self) -> str:
        if not self.run.dependencies:
            return ""
        rows = []
        for d in self.run.dependencies:
            pkg = d.get("package", "?")
            version = d.get("version", "?")
            path = d.get("path", "?")
            rows.append(f"""                <tr>
                    <td><code>{html.escape(pkg)}</code></td>
                    <td>{html.escape(version)}</td>
                    <td>{html.escape(str(path))}</td>
                </tr>""")
        return f"""    <div class="section">
        <h2>Dependencies ({len(self.run.dependencies)})</h2>
        <table>
            <thead><tr><th>Package</th><th>Version</th><th>Source</th></tr></thead>
            <tbody>
{chr(10).join(rows)}
            </tbody>
        </table>
    </div>"""

    def _build_vulnerabilities(self) -> str:
        if not self.run.vulnerabilities:
            return ""
        rows = []
        for v in self.run.vulnerabilities:
            severity = v.get("severity", "unknown").lower()
            try:
                sev_enum = Severity(severity)
                color = self.SEVERITY_COLORS.get(sev_enum, "#6b7280")
            except ValueError:
                color = "#6b7280"
            rows.append(f"""    <div class="vuln-row">
        <strong>{html.escape(v.get('package', '?'))}</strong> — {html.escape(v.get('id', ''))}
        <span class="severity-badge" style="background:{color}">{html.escape(severity.upper())}</span><br>
        <small>{html.escape(v.get('description', ''))}</small>
    </div>""")
        return f"""    <div class="section">
        <h2>Vulnerabilities ({len(self.run.vulnerabilities)})</h2>
        {chr(10).join(rows)}
    </div>"""
