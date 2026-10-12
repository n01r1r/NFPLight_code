"""Check training-policy navigation and provenance; no GPU, hashes, or training."""
import argparse
from datetime import datetime, timezone, timedelta
import json
from pathlib import Path
import re
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parent
BASELINE_DIRECTORY = "artifacts/policy_doctor_20261009"
ARTIFACT_KEYWORDS = ("bottleneck_fc", "regularization", "intermediate_capacity", "reduced_param", "scheduler_council", "model_capacity", "fabric_leather", "fabric_controlled_training", "fabric_degradation")
CONFIRMED_VALUES = {"default_scheduler": "ReduceLROnPlateau", "future_seed": 19, "run_seed31": False, "seed_scope": "current_reduced_param_followup", "precision": "FP32", "amp": False, "tf32": False, "near_samples": 4, "far_samples": 12, "gain": 9, "f0": 0.04, "lr_schedule_and_cap_extension_permitted": True}
LINK = re.compile(r"(?<!!)\[([^\]\n]+)\]\(([^)\n]+)\)")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def validate_policy(state):
    errors = []
    if not isinstance(state, dict):
        return ["policy must be an object"]
    if state.get("schema_version") != 1:
        errors.append("unsupported policy schema")
    confirmed = state.get("confirmed", {})
    for key in ("confirmed", "proposal", "implementation"):
        if not isinstance(state.get(key), dict):
            return [key + " must be an object"]
    for key, expected in CONFIRMED_VALUES.items():
        if type(confirmed.get(key)) is not type(expected) or confirmed[key] != expected:
            errors.append("confirmed contract mismatch: " + key)
    unknown = set(confirmed) - set(CONFIRMED_VALUES) - {"conditional_7m_request"}
    if unknown:
        errors.append("unconfirmed fields in confirmed policy: " + ",".join(sorted(unknown)))
    if not isinstance(confirmed.get("conditional_7m_request"), str):
        errors.append("conditional user request record missing")
    if confirmed.get("default_scheduler") != "ReduceLROnPlateau":
        errors.append("future default scheduler must be ReduceLROnPlateau")
    if confirmed.get("future_seed") != 19 or confirmed.get("run_seed31") is not False:
        errors.append("future single seed19 / excluded seed31 policy mismatch")
    proposal = state.get("proposal", {})
    if proposal.get("authority") != "council_recommendation_not_user_confirmation":
        errors.append("proposal incorrectly promoted to confirmed authority")
    leaked = set(confirmed) & (set(proposal) - {"authority"})
    if leaked:
        errors.append("unconfirmed proposal fields in confirmed policy: " + ",".join(sorted(leaked)))
    if proposal.get("floor_wait_consensus") is not False:
        errors.append("floor waiting time has no consensus")
    implementation = state.get("implementation", {})
    if implementation.get("state") == "pending":
        if implementation.get("default_applied_to_trainers") is not False or implementation.get("new_scheduler_training_started") is not False:
            errors.append("pending implementation cannot claim application/training")
    elif implementation.get("state") == "scoped_verified":
        if implementation.get("default_applied_to_trainers") is not False:
            errors.append("scoped implementation cannot claim all trainer defaults applied")
        if implementation.get("entrypoint") != "experiments/plateau_followup/run.py":
            errors.append("scoped scheduler entrypoint mismatch")
        if not isinstance(implementation.get("verification_record"), str):
            errors.append("scoped scheduler verification record missing")
        followup = state.get("current_followup", {})
        if followup.get("goal") != "validation NDR best map" or followup.get("contract") != "docs/PLATEAU_FOLLOWUP.md":
            errors.append("scoped scheduler followup goal/contract mismatch")
    else:
        errors.append("implementation state changed; update doctor only with verified implementation evidence")
    return errors


def validate_context(context):
    errors = []
    if context.get("schema_version") != 1:
        errors.append("unsupported artifact context schema")
    if context.get("policy_authority") is not False:
        errors.append("historical artifact/proposal cannot be current policy authority")
    if context.get("configuration_scope") != "historical_immutable":
        if context.get("configuration_scope") != "current_immutable" or context.get("authorized_contract") not in {"docs/PLATEAU_FOLLOWUP.md", "docs/ROBUSTNESS_SUITE.md"}:
            errors.append("artifact immutable configuration scope or current authorization missing")
    if context.get("canonical_policy") != "docs/TRAINING_SCHEDULER_POLICY.md":
        errors.append("artifact must point to canonical scheduler policy")
    return errors


def local_links(path):
    text = path.read_text(encoding="utf-8-sig")
    for label, target in LINK.findall(text):
        target = target.strip().strip("<>")
        if re.match(r"^(https?://|app://|codex://|mailto:|data:)", target):
            continue
        base, _, fragment = target.partition("#")
        base = re.sub(r":\d+$", "", unquote(base))
        absolute = bool(re.match(r"^[A-Za-z]:[/\\]", base))
        resolved = Path(base) if absolute else path.parent / base if base else path
        yield label, target, resolved.resolve(), unquote(fragment)


def heading_anchors(path):
    counts, anchors = {}, set()
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        match = re.match(r"^ {0,3}#{1,6} +(.+?)(?: +#+)?$", line)
        if not match:
            continue
        title = re.sub(r"<[^>]+>", "", match[1]).replace("`", "")
        slug = re.sub(r"[^\w\s-]", "", title.lower()).replace(" ", "-")
        count = counts.get(slug, 0)
        anchors.add(slug + ("-" + str(count) if count else ""))
        counts[slug] = count + 1
    return anchors


def scan_links(paths, root):
    checked, missing = [], []
    anchors = {}
    for path in paths:
        for label, target, resolved, fragment in local_links(path):
            record = dict(source=str(path.relative_to(root)), target=target, resolved=str(resolved), exists=resolved.exists())
            if record["exists"] and fragment and resolved.suffix == ".md" and not re.match(r"^L\d+(?:-L?\d+)?$", fragment):
                if resolved not in anchors:
                    anchors[resolved] = heading_anchors(resolved)
                record["anchor_exists"] = fragment in anchors[resolved]
            checked.append(record)
            if not record["exists"] or record.get("anchor_exists") is False:
                missing.append(record)
    return checked, missing


def changed_files(root, recorded):
    changed = []
    for record in recorded:
        path = Path(record["path"])
        if not path.is_absolute():
            path = root / path
        if not path.exists():
            changed.append(dict(record, reason="missing"))
            continue
        stat = path.stat()
        if stat.st_size != record["size"] or stat.st_mtime_ns != record["mtime_ns"]:
            changed.append(dict(record, reason="size/mtime changed"))
    return changed


def validate_index(root, index):
    errors = []
    base = (root / "artifacts").resolve()
    discovered = {str(p.relative_to(root)).replace("\\", "/") for p in base.iterdir() if p.is_dir() and (any(k in p.name for k in ARTIFACT_KEYWORDS) or (p / "artifact_context.json").exists())}
    registered = [entry["path"].replace("\\", "/") for entry in index["artifacts"]]
    if len(registered) != len(set(registered)):
        errors.append("duplicate artifact index entries")
    for path in sorted(discovered - set(registered)):
        errors.append("artifact omitted from index: " + path)
    for entry in index["artifacts"]:
        directory = (root / entry["path"]).resolve()
        if not directory.is_relative_to(base) or directory.parent != base or not directory.is_dir():
            errors.append("artifact index path outside root or missing: " + entry["path"])
        if (root / entry["context"]).resolve() != directory / "artifact_context.json":
            errors.append("artifact context belongs to another directory: " + entry["path"])
    return errors


def load_baselines(root):
    base = root / BASELINE_DIRECTORY
    required = ("baseline_scope.json", "preserved_files_before.json", "protected_identities.json")
    absent = [name for name in required if not (base / name).is_file()]
    if absent:
        return [], [], ["preservation baseline unavailable: " + ",".join(absent)]
    scope = read_json(base / required[0])
    preserved, identities = read_json(base / required[1]), read_json(base / required[2])
    errors = []
    for values, key in ((preserved, "preserved_files_count"), (identities, "frozen_identities_count")):
        if not values or len(values) != scope.get(key):
            errors.append("preservation baseline incomplete: " + key)
    if len({x["path"] for x in preserved}) != len(preserved):
        errors.append("duplicate preservation baseline paths")
    return preserved, identities, errors


def audit(root, output):
    errors, warnings = [], []
    state = read_json(root / "docs/TRAINING_POLICY_STATE.json")
    errors.extend(validate_policy(state))
    implementation = state["implementation"]
    if implementation.get("state") == "scoped_verified":
        evidence_path = root / implementation.get("verification_record", "missing.json")
        if not evidence_path.is_file():
            errors.append("scoped implementation verification artifact missing")
        else:
            evidence = read_json(evidence_path)
            if evidence.get("implementation_checks_passed") is not True or evidence.get("gpu_smoke_passed") is not True:
                errors.append("scoped implementation checks or GPU smoke not verified")
            if evidence.get("sha256_executed") is not False:
                errors.append("scoped implementation SHA-256 restriction missing")
        if not (root / implementation["entrypoint"]).is_file():
            errors.append("scoped implementation entrypoint missing")
    agents = (root / "AGENTS.md").read_text(encoding="utf-8-sig")
    if len(agents.splitlines()) > 9:
        errors.append("AGENTS.md exceeds nine lines")
    for required in ("docs/TRAINING_SCHEDULER_POLICY.md", "docs/CURRENT_TRAINING_STATE.md"):
        if required not in agents:
            errors.append("AGENTS missing mandatory policy navigation: " + required)
    documents = [root / "AGENTS.md", root / "README.md"]
    documents += sorted((root / "docs").rglob("*.md"))
    documents += sorted((root / "experiments").glob("*/README.md"))
    index = read_json(root / "artifacts/training_artifact_index.json")
    if index.get("artifact_role_is_not_policy_authority") is not True:
        errors.append("artifact index authority separation missing")
    errors.extend(validate_index(root, index))
    contexts = []
    for entry in index["artifacts"]:
        path = root / entry["context"]
        if not path.exists():
            errors.append("artifact context missing: " + entry["context"])
            continue
        context = read_json(path)
        if not isinstance(context, dict):
            errors.append("artifact context must be an object: " + entry["context"])
            continue
        contexts.append(context)
        if context.get("artifact") != entry["path"]:
            errors.append("artifact context identity mismatch: " + entry["path"])
        errors.extend(entry["path"] + ": " + e for e in validate_context(context))
        if context.get("role") != entry["role"]:
            errors.append("artifact index/context role mismatch: " + entry["path"])
        for key in ("canonical_policy", "current_training_state", "doctor_record"):
            if not (root / context[key]).exists():
                errors.append("context reference missing: " + context[key])
        directory = root / entry["path"]
        for p in directory.rglob("*.md"):
            # Frozen source snapshots inherit historical scope; never rewrite their source text.
            if any(part in ("source_snapshot", "source_snapshots", "legacy_code") for part in p.relative_to(directory).parts):
                continue
            documents.append(p)
    documents = sorted(set(documents))
    links, missing = scan_links(documents, root)
    errors.extend("broken local link: " + x["source"] + " -> " + x["target"] for x in missing)
    gate_documents = [root / "README.md", root / "docs/FABRIC_LEATHER_TRAINING.md", root / "docs/TRAINING_HTML_REPORT_CONTRACT.md"]
    gate_documents += list((root / "experiments").glob("*/README.md"))
    for p in gate_documents:
        text = p.read_text(encoding="utf-8-sig")
        if "TRAINING_SCHEDULER_POLICY.md" not in text or "CURRENT_TRAINING_STATE.md" not in text:
            errors.append("entry point missing policy/current-state links: " + str(p.relative_to(root)))
    regular = root / "artifacts/regularization_probe_20261009_checked"
    status, scope, conclusion = (read_json(regular / f) for f in ("status.json", "user_scope_completion.json", "two_seed_conclusion.json"))
    if status["status"] != "complete_selected_seeds" or status["selected_seeds"] != [7, 19] or status["scheduled_training_runs"] != 6:
        errors.append("regularization selected completion scope mismatch")
    if status["comparison_complete"] is not False or status["selected_comparison_complete"] is not True or status.get("pid") is not None:
        errors.append("regularization original/selected scope or stale live PID mismatch")
    incident = read_json(regular / "seed31_exclusion_incident.json")
    if scope["seed31_train_rows"] != 224 or incident["seed31_train_rows"] != 224 or not incident["excluded_from_comparison"]:
        errors.append("seed31 incident scope mismatch")
    if any(x["seed"] not in (7, 19) for x in conclusion["rows"]):
        errors.append("excluded seed leaked into selected comparison")
    seed31 = read_json(regular / "seed31_d4_only/status.json")
    if seed31["status"] != "interrupted_excluded" or not seed31["excluded_from_comparison"]:
        errors.append("excluded seed31 still appears active")
    capacity = root / "artifacts/intermediate_capacity_15m_seed19_20261009"
    completed, recovery, transition = (read_json(capacity / f) for f in ("complete.json", "export_recovery.json", "transition_status.json"))
    if completed["parameters"] != 15085034 or completed["map_updates"] != 10000 or completed["render_updates"] != 3000:
        errors.append("15M completed contract mismatch")
    if read_json(capacity / "status.json")["status"] != "complete" or recovery["training_updates_added"] != 0 or recovery["maximum_metric_difference"] != 0 or recovery["arrays"] != 32:
        errors.append("15M export recovery/completion mismatch")
    if transition["status"] != "completed_after_verified_manual_transition_and_export_recovery":
        errors.append("15M stale transition metadata")
    preserved, identities, baseline_errors = load_baselines(root)
    errors.extend(baseline_errors)
    preservation_changes = changed_files(root, preserved)
    errors.extend("preserved result/config changed: " + x["path"] for x in preservation_changes)
    mismatch = changed_files(root, identities)
    repair_path = root / "artifacts/ponytail_repairs_20261009/source_repairs.json"
    repairs = read_json(repair_path) if repair_path.exists() else []
    for x in mismatch:
        repair = next((r for r in repairs if Path(r["before"]["path"]).resolve() == Path(x["path"]).resolve()
                       and all(r["before"][k] == x[k] for k in ("size", "mtime_ns"))), None)
        if repair and (root / repair["original_archive"]).is_file() and not changed_files(root, [repair["after"]]):
            warnings.append("authorized source repair; historical lineage requires preserved source: " + x["path"])
            continue
        if x.get("source") == "artifacts\\regularization_probe_20261008\\execution_config.json":
            warnings.append("retired prepared artifact has pre-existing source identity mismatch; not reusable")
        else:
            errors.append("unexpected frozen identity mismatch: " + x["path"])
    warnings.append("Registered scheduler implementation scope: " + state["implementation"]["state"] + "; document PASS is not runtime or scientific validation")
    return dict(schema_version=1, status="pass_document_and_artifact_consistency" if not errors else "failed", checked_at_kst=datetime.now(timezone(timedelta(hours=9))).isoformat(), document_count=len(documents), local_link_count=len(links), artifact_roots=len(contexts), preserved_numeric_checkpoint_config_files=len(preserved), frozen_identity_records=len(identities), broken_local_links=missing, preservation_changes=preservation_changes, errors=errors, warnings=warnings, runtime_scheduler_state=state["implementation"], preservation_baseline=BASELINE_DIRECTORY, runtime_enforcement=False, claim_scope="Registered/discovered training artifact names and explicit context roots only; policy navigation, historical/proposed scope and size/mtime preservation only; no GPU, SHA-256, checkpoint loading, scientific validation or training")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "artifacts/policy_doctor_20261009")
    args = parser.parse_args()
    output = args.out.resolve(); output.mkdir(parents=True, exist_ok=True)
    # Context links point to this report even on the first invocation.
    report_path = output / "report.json"
    if not report_path.exists():
        report_path.write_text('{"status":"in_progress"}', encoding="utf-8")
    try:
        result = audit(ROOT, output)
    except (OSError, ValueError, TypeError, KeyError, AttributeError) as exc:
        # A malformed or missing record must replace any previous PASS.
        result = dict(schema_version=1, status="failed", document_count=0, local_link_count=0, artifact_roots=0, preserved_numeric_checkpoint_config_files=0, errors=["audit could not complete: " + str(exc)], warnings=["Incomplete audit; no preservation or runtime validation claimed"], runtime_enforcement=False)
    report_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = ["# 학습 정책 doctor", "", "판정: " + result["status"], "", f"문서 {result['document_count']}개, 로컬 link {result['local_link_count']}개, artifact root {result['artifact_roots']}개, 보존 파일 {result['preserved_numeric_checkpoint_config_files']}개 검사.", "", "## 발견·제한", ""]
    lines += ["- " + x for x in result["errors"] + result["warnings"]]
    baseline = (ROOT / BASELINE_DIRECTORY).as_posix()
    lines += ["", "## 변경·재현 근거", "", f"- [문서 변경과 보존 사본]({baseline}/document_changes.json)", f"- [Artifact navigation/metadata 변경]({baseline}/artifact_metadata_changes.json)", f"- [CPU 검증 기록]({baseline}/verification.json)", f"- [확정 정책]({ROOT.as_posix()}/docs/TRAINING_SCHEDULER_POLICY.md) · [현재 상태]({ROOT.as_posix()}/docs/CURRENT_TRAINING_STATE.md)", f"- [Artifact 인덱스]({ROOT.as_posix()}/artifacts/training_artifact_index.json)", "", "재실행: `python doctor_training_policy.py`; 테스트: `python -m pytest tests/test_training_policy_doctor.py -q`.", "", "확정/제안/이력 연결과 metadata를 보정했다. 원래 source snapshot과 보존 사본은 역사 기록으로 유지한다. 새 default 구현·학습·성능 개선을 인증하지 않는다. 기존 수치·checkpoint·immutable config의 size/mtime 보존만 검사했으며 SHA-256은 실행하지 않았다.", ""]
    (output / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("status", "document_count", "local_link_count", "artifact_roots", "preserved_numeric_checkpoint_config_files", "errors", "warnings")}, ensure_ascii=False))
    return 1 if result["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
