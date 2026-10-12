"""Build saved summaries and plots from one completed or partial run directory."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.bottleneck_fc.config import atomic_write_json, load_execution_config, resolve_repo_path


def build_report(run_dir):
    run_dir = Path(run_dir).resolve()
    execution = load_execution_config(run_dir)
    settings = execution["settings"]
    arms = settings["model"]["arms"]
    labels = {arm["id"]: arm["label"] for arm in arms}
    colors = {arm["id"]: arm["color"] for arm in arms}
    loaded = {}
    complete = {}
    for seed in settings["training"]["seeds"]:
        for arm in arms:
            run_path = run_dir / f"seed{seed}_{arm['id']}"
            evaluation_path = run_path / "evaluation.json"
            complete_path = run_path / "complete.json"
            if evaluation_path.is_file():
                loaded[seed, arm["id"]] = json.loads(evaluation_path.read_text(encoding="utf-8"))
            if complete_path.is_file():
                complete[seed, arm["id"]] = json.loads(complete_path.read_text(encoding="utf-8"))

    summary = {
        "completed_runs": len(complete),
        "expected_runs": len(settings["training"]["seeds"]) * len(arms),
        "history_count": len(loaded),
        "rows": [],
        "claim": "bounded dynamics and same-renderer relighting; no assumption of convergence",
    }
    for (seed, arm_id), data in complete.items():
        summary["rows"].append({
            "seed": seed,
            "arm_id": arm_id,
            "label": labels[arm_id],
            "parameters": data["parameters"],
            **{population: data["final"][population]["mean"] for population in data["final"]},
        })

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from PIL import Image, ImageDraw
    from capture_processing.difference_visualization import (
        display_range,
        field_image,
        map_difference,
        render_delta_e00,
    )

    if loaded:
        figure, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True)
        for column, population in enumerate(("train", "unseen_crop", "heldout")):
            for row_index, metric in enumerate(("ndr_mean", "render_l1")):
                axis = axes[row_index, column]
                for (seed, arm_id), history in loaded.items():
                    axis.plot(
                        [item["total_step"] for item in history],
                        [item["populations"][population]["mean"][metric] for item in history],
                        color=colors[arm_id],
                        alpha=0.75,
                        label=f"{labels[arm_id]} seed{seed}",
                    )
                axis.set_title(f"{population}: {metric}")
                axis.grid(alpha=0.25)
                axis.set_xlabel("Optimizer update")
                axis.set_ylabel("L1")
                axis.axvline(
                    settings["training"]["map_updates"],
                    color="gray",
                    linestyle="--",
                    alpha=0.5,
                )
        axes[0, 0].legend(fontsize=8)
        figure.suptitle("Paired bottleneck probe — recorded evaluations; dashed line: render stage")
        figure.tight_layout()
        figure.savefig(run_dir / "dynamics.png", dpi=140)
        plt.close(figure)

    if len(complete) == summary["expected_runs"]:
        effects = {}
        for reference in ("baseline", "pixel_mlp"):
            effects[reference] = {}
            for population in ("train", "unseen_crop", "heldout"):
                effects[reference][population] = {}
                for key in ("ndr_mean", "normal_degrees", "render_l1", "near_l1"):
                    paired = []
                    relative = []
                    for seed in settings["training"]["seeds"]:
                        grouped = []
                        for arm_id in ("bottleneck_fc", reference):
                            by_material = {}
                            for row in complete[seed, arm_id]["final"][population]["materials"]:
                                by_material.setdefault(row["material_id"], []).append(row[key])
                            grouped.append({
                                material: float(np.mean(values))
                                for material, values in by_material.items()
                            })
                        material_ids = sorted(grouped[0])
                        if material_ids != sorted(grouped[1]):
                            raise ValueError(f"unpaired materials for seed {seed}, {population}")
                        paired.append([
                            grouped[0][material] - grouped[1][material]
                            for material in material_ids
                        ])
                        relative.append(
                            (np.mean(list(grouped[0].values())) /
                             np.mean(list(grouped[1].values())) - 1) * 100
                        )
                    values = np.asarray(paired)
                    per_material = values.mean(axis=0)
                    rng = np.random.default_rng(991)
                    bootstrap = per_material[
                        rng.integers(0, len(per_material), (10000, len(per_material)))
                    ].mean(axis=1)
                    effects[reference][population][key] = {
                        "bottleneck_fc_minus_reference": float(values.mean()),
                        "relative_percent_by_seed": relative,
                        "paired_seed_mean_differences": values.mean(axis=1).tolist(),
                        "conditional_material_bootstrap95": np.percentile(
                            bootstrap, [2.5, 97.5]
                        ).tolist(),
                        "confidence_limit": (
                            "material resampling conditional on three fixed seeds; "
                            "not initialization uncertainty or full-population confirmation"
                        ),
                    }
        summary["effects"] = effects

        first_seed = settings["training"]["seeds"][0]
        samples = sorted(
            (run_dir / f"seed{first_seed}_baseline" / "arrays").glob("train_*.npz")
        )[:2]
        for sample in samples:
            fields, rgb = {}, {}
            for arm in arms:
                arm_id = arm["id"]
                with np.load(
                    run_dir / f"seed{first_seed}_{arm_id}" / "arrays" / sample.name,
                    allow_pickle=False,
                ) as saved:
                    prediction = np.moveaxis(saved["prediction"][0], 0, -1)
                    target = np.moveaxis(saved["target"][0], 0, -1)
                    valid = saved["valid"][0, 0].astype(bool)
                    near_gt = np.moveaxis(saved["near_gt"][0], 0, -1)
                    near_prediction = np.moveaxis(saved["near_prediction"][0], 0, -1)
                    fields[arm_id] = {
                        "normal": map_difference(target[:, :, :3], prediction[:, :, :3], "normal"),
                        "roughness": (
                            (prediction[:, :, 6] - target[:, :, 6]) / 2
                        ),
                        "render": render_delta_e00(near_gt, near_prediction),
                    }
                    rgb[arm_id] = np.uint8(
                        np.clip(near_prediction, 0, 1) ** (1 / 2.2) * 255
                    )
            scales = {
                key: display_range(
                    [fields[arm["id"]][key][valid] for arm in arms],
                    signed=key == "roughness",
                )
                for key in ("normal", "roughness", "render")
            }
            image = Image.new("RGB", (1024, len(arms) * 375), "white")
            draw = ImageDraw.Draw(image)
            for index, arm in enumerate(arms):
                arm_id = arm["id"]
                top = index * 375
                draw.text((5, top + 2), f"{arm['label']}; {sample.stem}", fill="black")
                image.paste(Image.fromarray(rgb[arm_id]), (0, top + 20))
                for column, (key, label) in enumerate((
                    ("normal", "Normal angle (deg)"),
                    ("roughness", "Roughness signed pred-GT"),
                    ("render", "Near render DeltaE00"),
                )):
                    values = fields[arm_id][key].copy()
                    values[~valid] = 0
                    tile = field_image(values, scales[key], label)
                    image.paste(tile, ((column + 1) * 256, top + 20))
            image.save(run_dir / f"comparison_{sample.stem}.png")

    atomic_write_json(run_dir / "summary.json", summary)
    print(json.dumps({
        "completed": len(complete),
        "expected": summary["expected_runs"],
        "available_histories": len(loaded),
        "summary": str(run_dir / "summary.json"),
    }, ensure_ascii=False))
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    return build_report(resolve_repo_path(args.run_dir))


if __name__ == "__main__":
    main()
