import base64
import os
import subprocess
from typing import Any


PROJECT_ID = os.environ.get("PROJECT_ID", "").strip() or os.environ.get("GCP_PROJECT", "").strip()
REGION = os.environ.get("REGION", "").strip() or "us-central1"
SERVICE = os.environ.get("SERVICE", "").strip() or "dama"


def budget_kill_chat(event: dict[str, Any], context: Any) -> None:
    """
    Pub/Sub-triggered function. When a Billing Budget threshold is reached,
    disable DAMA chat by setting DAMA_DISABLE_CHAT=1 on the Cloud Run service.
    """
    # Decode payload if present (not required for the action).
    try:
        data = event.get("data")
        if isinstance(data, str) and data:
            base64.b64decode(data)
    except Exception:
        pass

    if not PROJECT_ID:
        raise RuntimeError("PROJECT_ID env var required")

    subprocess.check_call(
        [
            "gcloud",
            "run",
            "services",
            "update",
            SERVICE,
            "--project",
            PROJECT_ID,
            "--region",
            REGION,
            "--set-env-vars",
            "DAMA_DISABLE_CHAT=1",
            "--quiet",
        ]
    )

