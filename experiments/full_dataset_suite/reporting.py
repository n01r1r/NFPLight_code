"""Translate full-population evaluations for the existing live plot template."""
from experiments.regularization.config import atomic_write_json


def inference_payload(selected, execution):
    """Keep feature geometry and backbone metadata in model-only exports."""
    from copy import deepcopy
    rendering = execution["scientific_settings"]["rendering"]
    return {"net_g": selected["net_g"],
            "feature_version": rendering["feature_version"],
            "relation_version": rendering["relation_version"],
            "backbone": deepcopy(execution["scientific_settings"]["backbone"]),
            "precision": "float32", "normal_decoder": "unit XYZ"}


def write_offline_report(output):
    """Freeze the existing viewer with its final recorded state, including pauses."""
    import json
    from pathlib import Path
    from live_training_plot import HTML_PATH, LogReader
    output = Path(output)
    snapshot = LogReader(output.resolve()).snapshot()
    atomic_write_json(output / "offline_snapshot.json", snapshot)
    payload = json.dumps(snapshot, allow_nan=False).replace("<", "\\u003c")
    html = HTML_PATH.read_text(encoding="utf-8")
    fetch = "await fetch('/api',{cache:'no-store'})"
    timer = "setInterval(refresh,10000);"
    if html.count(fetch) != 1 or html.count(timer) != 1:
        raise ValueError("live template changed; offline substitution requires review")
    html = html.replace(fetch, "({ok:true,json:async()=>(" + payload + ")})")
    html = html.replace(timer, "")
    (output / "offline_report.html").write_text(html, encoding="utf-8")

def write_history(output, history, settings):
    rows = []
    for entry in history:
        heldout = entry["populations"]["heldout"]
        rows.append({"stage": entry["stage"], "step": entry["step"],
                     "total_step": entry["total_step"], **heldout["mean"],
                     "category": heldout["category"], "materials": heldout["materials"],
                     "population": "all164 validation sample folders, four fixed crops, novel12 lights"})
    atomic_write_json(output / "validation.json", {"history": rows})
