import asyncio
from datetime import datetime

from fastapi import BackgroundTasks, FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse

from agent import analyze_and_fix_failure

app = FastAPI(title="Pipeline Fixer")

# In-memory log — reset on app restart. Replace with a Delta table write in
# agent.py if you need persistence across restarts.
fix_log: list[dict] = []


@app.post("/webhook")
async def webhook(request: Request, background_tasks: BackgroundTasks):
    """Receives Databricks jobs.on_failure webhook payloads."""
    payload = await request.json()

    event_type = payload.get("event_type", "")
    if event_type != "jobs.on_failure":
        return JSONResponse({"status": "ignored", "event_type": event_type})

    run_id = int(payload["run_id"])
    job_name = payload.get("job_name", str(payload.get("job_id", "unknown")))

    entry = {
        "run_id": run_id,
        "job_name": job_name,
        "started_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "status": "analyzing",
        "root_cause": None,
        "fix_applied": False,
        "confidence": None,
        "repair_run_id": None,
    }
    fix_log.insert(0, entry)

    background_tasks.add_task(_run_fixer, entry, run_id)
    return JSONResponse({"status": "queued", "run_id": run_id})


async def _run_fixer(entry: dict, run_id: int):
    result = await asyncio.to_thread(analyze_and_fix_failure, run_id)
    if asyncio.iscoroutine(result):
        result = await result
    entry.update(result)


@app.get("/", response_class=HTMLResponse)
async def dashboard():
    """Simple status dashboard showing recent fix attempts."""
    rows = ""
    for e in fix_log[:50]:
        color = {
            "analyzing": "#f0a500",
            "fixed": "#2e7d32",
            "low_confidence": "#616161",
            "failed": "#c62828",
        }.get(e["status"], "#1565c0")

        root_cause = (e.get("root_cause") or "")[:120]
        repair = e.get("repair_run_id") or "—"
        confidence = f"{e['confidence']:.0%}" if e.get("confidence") is not None else "—"

        rows += f"""
        <tr>
          <td>{e['run_id']}</td>
          <td>{e['job_name']}</td>
          <td style="color:{color};font-weight:bold">{e['status']}</td>
          <td>{confidence}</td>
          <td>{root_cause}</td>
          <td>{'Yes' if e.get('fix_applied') else 'No'}</td>
          <td>{repair}</td>
          <td>{e['started_at']}</td>
        </tr>"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta http-equiv="refresh" content="30">
  <title>Pipeline Fixer</title>
  <style>
    body {{ font-family: monospace; padding: 24px; background: #fafafa; }}
    h1 {{ margin-bottom: 4px; }}
    p.sub {{ color: #555; margin-top: 0; font-size: 0.9em; }}
    table {{ border-collapse: collapse; width: 100%; background: white; box-shadow: 0 1px 3px rgba(0,0,0,.12); }}
    th {{ background: #263238; color: white; padding: 10px 12px; text-align: left; }}
    td {{ padding: 8px 12px; border-bottom: 1px solid #e0e0e0; vertical-align: top; }}
    tr:hover {{ background: #f5f5f5; }}
  </style>
</head>
<body>
  <h1>Pipeline Fixer</h1>
  <p class="sub">Auto-refreshes every 30 s &mdash; showing last {len(fix_log)} event(s)</p>
  <table>
    <tr>
      <th>Run ID</th><th>Job</th><th>Status</th><th>Confidence</th>
      <th>Root Cause</th><th>Fixed</th><th>Repair Run</th><th>Time (UTC)</th>
    </tr>
    {rows if rows else '<tr><td colspan="8" style="text-align:center;color:#777;padding:24px">No events yet</td></tr>'}
  </table>
</body>
</html>"""
    return html


@app.get("/health")
async def health():
    return {"status": "ok", "events": len(fix_log)}
