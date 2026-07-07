import base64
import json
import os
from datetime import date

import httpx
from anthropic import Anthropic
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.jobs import RunResultState
from databricks.sdk.service.workspace import ImportFormat, Language

anthropic_client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
w = WorkspaceClient()

# Map task key fragment → which API docs to fetch
NOTEBOOK_DOCS_MAP = {
    "stage_stocks": "yfinance",
    "stage_news": "yfinance",
    "stage_fx": "frankfurter",
    "stage_macro": "ecb",
}

API_DOCS_URLS = {
    "yfinance": "https://raw.githubusercontent.com/ranaroussi/yfinance/main/README.md",
    "frankfurter": "https://www.frankfurter.app/docs",
    "ecb": "https://data.ecb.europa.eu/help/api/overview",
}

SYSTEM_PROMPT = """You are a Databricks pipeline repair agent. You will receive:
1. An error traceback from a failed Databricks notebook task
2. The full Python source of that notebook
3. Documentation for the external API the notebook uses

Respond with ONLY a JSON object (no markdown, no explanation outside the JSON):
{
  "root_cause": "<one-sentence explanation of what caused the error>",
  "fixed_code": "<the full corrected Python source with only the minimal fix applied>",
  "confidence": <float 0.0–1.0>,
  "explanation": "<what you changed and why>"
}

Rules:
- Only fix the actual error. Do not refactor, rename, add features, or change anything unrelated.
- If you cannot confidently determine the fix, set confidence below 0.8 and leave fixed_code identical to the input source.
- Prefer minimal, targeted changes: add a retry, fix a URL, handle a None — nothing more."""


async def analyze_and_fix_failure(run_id: int) -> dict:
    try:
        run = w.jobs.get_run(run_id=run_id)
        failed_task = _find_failed_task(run)
        if not failed_task:
            return {"status": "failed", "root_cause": "No failed task found in run"}

        task_key = failed_task.task_key
        notebook_path = (
            failed_task.notebook_task.notebook_path
            if failed_task.notebook_task
            else None
        )
        if not notebook_path:
            return {"status": "failed", "root_cause": f"Task {task_key} has no notebook"}

        task_run_id = failed_task.run_id
        error_trace = _get_error(task_run_id)
        notebook_source = _read_notebook(notebook_path)

        api_name = next(
            (v for k, v in NOTEBOOK_DOCS_MAP.items() if k in task_key), None
        )
        docs = _fetch_docs(api_name) if api_name else ""

        result = _call_claude(error_trace, notebook_source, docs)
        confidence = float(result.get("confidence", 0))
        root_cause = result.get("root_cause", "Unknown")
        fixed_code = result.get("fixed_code", "")

        if confidence < 0.8 or not fixed_code or fixed_code == notebook_source:
            return {
                "status": "low_confidence",
                "root_cause": root_cause,
                "fix_applied": False,
                "confidence": confidence,
            }

        backup_path = notebook_path + "_backup_" + date.today().isoformat().replace("-", "")
        _write_notebook(backup_path, notebook_source)
        _write_notebook(notebook_path, fixed_code)

        repair = w.jobs.repair_run(
            run_id=run_id,
            rerun_tasks=[task_key],
        )

        return {
            "status": "fixed",
            "root_cause": root_cause,
            "fix_applied": True,
            "confidence": confidence,
            "explanation": result.get("explanation", ""),
            "repair_run_id": repair.repair_run_id,
            "backup_path": backup_path,
        }

    except Exception as exc:
        return {"status": "failed", "root_cause": str(exc), "fix_applied": False}


def _find_failed_task(run):
    for task in run.tasks or []:
        state = task.state
        if state and state.result_state == RunResultState.FAILED:
            return task
    return None


def _get_error(task_run_id: int) -> str:
    try:
        output = w.jobs.get_run_output(run_id=task_run_id)
        parts = [output.error or "", output.error_trace or ""]
        return "\n".join(p for p in parts if p).strip()
    except Exception as exc:
        return f"Could not retrieve run output: {exc}"


def _read_notebook(path: str) -> str:
    export = w.workspace.export(path=path, format="SOURCE")
    if export.content:
        return base64.b64decode(export.content).decode("utf-8")
    return ""


def _write_notebook(path: str, source: str) -> None:
    encoded = base64.b64encode(source.encode("utf-8")).decode("utf-8")
    w.workspace.import_(
        path=path,
        content=encoded,
        format=ImportFormat.SOURCE,
        language=Language.PYTHON,
        overwrite=True,
    )


def _fetch_docs(api_name: str) -> str:
    url = API_DOCS_URLS.get(api_name, "")
    if not url:
        return ""
    try:
        r = httpx.get(url, timeout=15, follow_redirects=True)
        return r.text[:8000]
    except Exception:
        return ""


def _call_claude(error: str, source: str, docs: str) -> dict:
    user_message = (
        f"ERROR TRACEBACK:\n{error}\n\n"
        f"NOTEBOOK SOURCE:\n{source}\n\n"
        f"API DOCUMENTATION:\n{docs}"
    )
    response = anthropic_client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=4096,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    )
    text = response.content[0].text.strip()
    start = text.find("{")
    end = text.rfind("}") + 1
    return json.loads(text[start:end])
