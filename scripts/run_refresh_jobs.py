from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from streamlit_app import load_refresh_jobs, run_due_refresh_jobs, save_refresh_jobs


def main() -> None:
    jobs = load_refresh_jobs()
    if not jobs:
        print("No refresh jobs configured. Nothing to run.")
        return

    updated_jobs, logs = run_due_refresh_jobs(jobs)
    save_refresh_jobs(updated_jobs)

    print(f"[{datetime.now(timezone.utc).isoformat()}] Refresh runner complete")
    for line in logs:
        print(f"- {line}")

    summary = {
        "job_count": len(updated_jobs),
        "logs": logs,
    }
    Path(".datavet/last_refresh_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
