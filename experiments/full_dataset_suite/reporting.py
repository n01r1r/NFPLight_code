"""Translate full-population evaluations for the existing live plot template."""
from experiments.regularization.config import atomic_write_json

def write_history(output, history, settings):
    rows = []
    for entry in history:
        heldout = entry["populations"]["heldout"]
        rows.append({"stage": entry["stage"], "step": entry["step"],
                     "total_step": entry["total_step"], **heldout["mean"],
                     "category": heldout["category"], "materials": heldout["materials"],
                     "population": "all164 validation sample folders, four fixed crops, novel12 lights"})
    atomic_write_json(output / "validation.json", {"history": rows})
