import copy
import json
import pytest
import doctor_training_policy as doctor
from pathlib import Path

from doctor_training_policy import (
    ROOT, changed_files, read_json, scan_links, validate_context, validate_policy,
)


def policy():
    return copy.deepcopy(read_json(ROOT / "docs/TRAINING_POLICY_STATE.json"))


def test_current_policy_is_consistent_without_claiming_runtime_implementation():
    state = policy()
    assert validate_policy(state) == []
    assert not state["implementation"]["default_applied_to_trainers"]
    assert state["proposal"]["floor_wait_consensus"] is False


def test_rejects_council_proposal_promoted_to_user_decision():
    state = policy()
    state["proposal"]["authority"] = "confirmed_user_decision"
    state["confirmed"]["monitor"] = state["proposal"]["monitor"]
    errors = validate_policy(state)
    assert any("promoted" in error for error in errors)
    assert any("unconfirmed" in error for error in errors)


def test_rejects_legacy_schedule_and_seed_leaking_into_future_default():
    state = policy()
    state["confirmed"].update(default_scheduler="cosine", run_seed31=True)
    errors = validate_policy(state)
    assert any("scheduler" in error for error in errors)
    assert any("seed31" in error for error in errors)


def test_rejects_pending_implementation_claimed_as_started():
    state = policy()
    state["implementation"]["state"] = "pending"
    state["implementation"]["new_scheduler_training_started"] = True
    assert any("pending" in error for error in validate_policy(state))


def test_rejects_historical_artifact_used_as_policy_authority():
    entry = read_json(ROOT / "artifacts/training_artifact_index.json")["artifacts"][0]
    context = read_json(ROOT / entry["context"])
    assert validate_context(context) == []
    context["policy_authority"] = True
    assert any("authority" in error for error in validate_context(context))


def test_detects_missing_local_file_and_heading_without_fetching_external_urls(tmp_path):
    doc = tmp_path / "README.md"
    target = tmp_path / "policy.md"
    target.write_text("# Current Policy\n", encoding="utf-8")
    doc.write_text("[good](policy.md#current-policy)\n[bad heading](policy.md#nonexistent)\n[missing](gone.json)\n[external](https://example.invalid/no-fetch)\n", encoding="utf-8")
    checked, missing = scan_links([doc], tmp_path)
    assert len(checked) == 3
    assert {x["target"] for x in missing} == {"policy.md#nonexistent", "gone.json"}


def test_preservation_check_detects_changed_config(tmp_path):
    path = tmp_path / "config.json"
    path.write_text('{"seed":19}', encoding="utf-8")
    stat = path.stat()
    records = [dict(path="config.json", size=stat.st_size, mtime_ns=stat.st_mtime_ns)]
    assert changed_files(tmp_path, records) == []
    path.write_text('{"seed":31,"modified":true}', encoding="utf-8")
    assert changed_files(tmp_path, records)[0]["reason"] == "size/mtime changed"


@pytest.mark.parametrize("key,value", [("precision", "FP16"), ("amp", True), ("tf32", True), ("near_samples", 12), ("gain", 4), ("f0", 0.2), ("seed_scope", "all_future_research")])
def test_rejects_physical_contract_or_seed_scope_contamination(key, value):
    state = policy()
    state["confirmed"][key] = value
    assert any(key in error for error in validate_policy(state))


def test_rejects_disguised_nested_approval_outside_confirmed_schema():
    state = policy()
    state["confirmed"]["approved_scheduler_config"] = state["proposal"]
    assert any("approved_scheduler_config" in error for error in validate_policy(state))


def test_report_output_does_not_disable_preservation_checks(tmp_path, monkeypatch):
    """Audit a tiny fixture tree; production cleanup history is not a unit fixture."""
    root=tmp_path / "repo";root.mkdir()
    def write(relative,value):
        path=root / relative;path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(value) if isinstance(value,dict) else value,encoding="utf8")
    state=policy()
    write("docs/TRAINING_POLICY_STATE.json",state)
    links="[policy](docs/TRAINING_SCHEDULER_POLICY.md) [state](docs/CURRENT_TRAINING_STATE.md)"
    write("AGENTS.md",links);write("README.md",links)
    for name in ("TRAINING_SCHEDULER_POLICY.md","CURRENT_TRAINING_STATE.md","FABRIC_LEATHER_TRAINING.md","TRAINING_HTML_REPORT_CONTRACT.md"):
        write("docs/"+name,"TRAINING_SCHEDULER_POLICY.md CURRENT_TRAINING_STATE.md")
    implementation=state["implementation"]
    write(implementation["entrypoint"],"# test fixture")
    write(implementation["verification_record"],{"implementation_checks_passed":True,"gpu_smoke_passed":True,"sha256_executed":False})
    write("artifacts/training_artifact_index.json",{"artifact_role_is_not_policy_authority":True,"artifacts":[]})
    # Index behavior has separate tests; this test exercises audit preservation.
    monkeypatch.setattr(doctor,"validate_index",lambda *args:[])
    for directory,names in (
        ("regularization_probe_20261009_checked",("status.json","user_scope_completion.json","two_seed_conclusion.json","seed31_exclusion_incident.json","seed31_d4_only/status.json")),
        ("intermediate_capacity_15m_seed19_20261009",("complete.json","export_recovery.json","transition_status.json","status.json")),
    ):
        for name in names:write("artifacts/"+directory+"/"+name,read_json(ROOT / "artifacts" / directory / name))
    protected=root / "record.json";protected.write_text('{"step":1}',encoding="utf8")
    st=protected.stat()
    original=[{"path":str(protected),"size":st.st_size,"mtime_ns":st.st_mtime_ns}]
    monkeypatch.setattr(doctor,"load_baselines",lambda *args:(original,[],[]))
    result=doctor.audit(root,root)
    assert result["errors"]==[]
    assert result["preserved_numeric_checkpoint_config_files"]==1
    assert result["runtime_enforcement"] is False
    protected.write_text('{"step":123456}',encoding="utf8")
    changed=doctor.audit(root,root)
    assert any("preserved result/config changed" in x for x in changed["errors"])
    assert len(changed["preservation_changes"])==1


def test_omitted_artifact_is_detected_independently_of_index():
    index = read_json(ROOT / "artifacts/training_artifact_index.json")
    index["artifacts"] = [x for x in index["artifacts"] if "scheduler_council" not in x["path"]]
    assert any("omitted" in e and "scheduler_council" in e for e in doctor.validate_index(ROOT, index))


def test_missing_or_truncated_baselines_cannot_claim_preservation(tmp_path):
    assert doctor.load_baselines(tmp_path)[2]
    base = tmp_path / doctor.BASELINE_DIRECTORY
    base.mkdir(parents=True)
    (base / "baseline_scope.json").write_text(json.dumps(dict(preserved_files_count=1, frozen_identities_count=1)), encoding="utf-8")
    for filename in ("preserved_files_before.json", "protected_identities.json"):
        (base / filename).write_text("[]", encoding="utf-8")
    assert len(doctor.load_baselines(tmp_path)[2]) == 2


def test_artifact_path_cannot_escape_artifacts_root(tmp_path):
    (tmp_path / "artifacts").mkdir()
    (tmp_path / "docs").mkdir()
    index = {"artifacts": [dict(path="docs", context="docs/artifact_context.json")]}
    assert any("outside" in e for e in doctor.validate_index(tmp_path, index))


def test_incomplete_audit_replaces_previous_pass(tmp_path, monkeypatch):
    output = tmp_path / "report"
    output.mkdir()
    (output / "report.json").write_text('{"status":"pass_document_and_artifact_consistency"}', encoding="utf-8")
    monkeypatch.setattr(doctor, "ROOT", tmp_path)
    monkeypatch.setattr("sys.argv", ["doctor", "--out", str(output)])
    assert doctor.main() == 1
    assert read_json(output / "report.json")["status"] == "failed"


def test_malformed_context_replaces_previous_pass(tmp_path, monkeypatch):
    original = doctor.read_json
    monkeypatch.setattr(doctor, 'read_json', lambda path: [] if path.name == 'artifact_context.json' else original(path))
    (tmp_path / 'report.json').write_text('{"status":"pass_document_and_artifact_consistency"}')
    monkeypatch.setattr('sys.argv', ['doctor', '--out', str(tmp_path)])
    assert doctor.main() == 1
    assert original(tmp_path / 'report.json')['status'] == 'failed'


def test_new_training_is_blocked_before_output_or_cuda(tmp_path):
    import subprocess, sys
    code = "from train_fabric import require_scheduler_ready,run,parse_args; require_scheduler_ready(historical_resume=True); require_scheduler_ready(smoke=True); run(parse_args(['--out'," + repr(str(tmp_path / 'fresh')) + "]))"
    result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True)
    assert result.returncode != 0 and 'ReduceLROnPlateau' in result.stderr
    assert not (tmp_path / 'fresh').exists()


def test_scoped_scheduler_requires_entrypoint_verification_and_goal():
    state = policy()
    state["implementation"].update(state="scoped_verified", default_applied_to_trainers=False,
        entrypoint="experiments/plateau_followup/run.py", verification_record="artifacts/plateau_followup_verification_20261009/verification.json")
    state["current_followup"] = {"goal": "validation NDR best map", "contract": "docs/PLATEAU_FOLLOWUP.md"}
    assert validate_policy(state) == []
    state["implementation"]["default_applied_to_trainers"] = True
    assert any("all trainer" in error for error in validate_policy(state))
    state["implementation"]["default_applied_to_trainers"] = False
    state["current_followup"]["goal"] = "train only"
    assert any("goal" in error for error in validate_policy(state))


def test_current_artifact_scope_requires_authorized_contract():
    context = {"schema_version":1,"policy_authority":False,"canonical_policy":"docs/TRAINING_SCHEDULER_POLICY.md",
        "configuration_scope":"current_immutable","authorized_contract":"docs/PLATEAU_FOLLOWUP.md"}
    assert validate_context(context) == []
    context.pop("authorized_contract")
    assert any("authorization" in error for error in validate_context(context))
