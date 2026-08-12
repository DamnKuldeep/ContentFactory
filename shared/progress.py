"""
Content Factory — Live Progress Dashboard using Rich.
"""

import time
from rich.live import Live
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich.layout import Layout

from .manifest import Manifest

def render_dashboard(manifest: Manifest, batch_id: str, stage: str, total: int) -> Layout:
    stats = manifest.get_stats(batch_id, stage)
    
    pending = stats.get('PENDING', 0)
    running = stats.get('RUNNING', 0)
    complete = stats.get('COMPLETE', 0)
    failed = stats.get('FAILED', 0)
    upload = stats.get('UPLOAD_PENDING', 0)
    
    # Create the summary table
    table = Table(show_header=True, header_style="bold magenta", expand=True)
    table.add_column("Status", style="dim", width=15)
    table.add_column("Count", justify="right")
    table.add_column("Percentage", justify="right")
    
    def pct(val):
        return f"{(val/total)*100:.1f}%" if total > 0 else "0%"
        
    table.add_row("[cyan]PENDING", str(pending), pct(pending))
    table.add_row("[yellow]RUNNING", str(running), pct(running))
    table.add_row("[green]COMPLETE", str(complete), pct(complete))
    if upload > 0:
        table.add_row("[blue]UPLOADING", str(upload), pct(upload))
    if failed > 0:
        table.add_row("[red]FAILED", str(failed), pct(failed))
        
    # Error panel
    failures = manifest.get_failed(batch_id, stage)
    if failures:
        err_text = Text()
        for f in failures[-3:]: # show last 3
            err_text.append(f"Story {f['story_num']} (Attempt {f['attempts']}): ", style="bold red")
            err_msg = (f['last_error'] or "").split('\n')[-1][:100]
            err_text.append(f"{err_msg}\n")
        err_panel = Panel(err_text, title="[red]Recent Errors", border_style="red")
    else:
        err_panel = Panel(Text("No errors.", style="green"), title="Errors", border_style="green")
        
    layout = Layout()
    layout.split_column(
        Layout(Panel(table, title=f"[bold]Batch {batch_id} ({stage}) Progress ({total} total)", border_style="blue")),
        Layout(err_panel)
    )
    return layout

def monitor_progress(db_path: str, batch_id: str, stage: str, total: int):
    """Run a live Rich display until all jobs are complete."""
    manifest = Manifest(db_path)
    
    with Live(render_dashboard(manifest, batch_id, stage, total), refresh_per_second=1) as live:
        while True:
            time.sleep(1)
            live.update(render_dashboard(manifest, batch_id, stage, total))
            if not manifest.has_pending_work(batch_id, stage):
                break
