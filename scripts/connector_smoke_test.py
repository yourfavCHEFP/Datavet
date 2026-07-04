from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from streamlit_app import connector_health_check


def main() -> None:
    rows = connector_health_check("machine learning")
    if not rows:
        raise RuntimeError("No connector health rows returned.")

    print("Connector smoke test summary:")
    for row in rows:
        print(
            f"- {row['connector']}: status={row['status']} latency_ms={row['latency_ms']} sample_count={row['sample_count']}"
        )

    fail_count = sum(1 for row in rows if row.get("status") == "fail")
    if fail_count == len(rows):
        raise RuntimeError("All connectors reported failure. Check network or provider changes.")


if __name__ == "__main__":
    main()
