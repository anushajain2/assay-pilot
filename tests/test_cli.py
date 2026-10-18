"""Command-line behaviour, including the exit codes a script would branch on."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from assaypilot.cli import main


@pytest.fixture
def inputs(tmp_path, simulated):
    """Write the simulated campaign to disk and return the paths."""
    directory = tmp_path / "inputs"
    paths = simulated.write(directory)
    return {k: str(v) for k, v in paths.items()}


def test_init_campaign_writes_a_definition(tmp_path, reference, capsys):
    output = tmp_path / "campaign.json"
    code = main(
        [
            "init-campaign",
            "--id",
            "demo",
            "--reference",
            reference,
            "--assay-name",
            "initial rate",
            "--units",
            "s^-1",
            "--condition",
            "37C",
            "--batch-size",
            "8",
            "-o",
            str(output),
        ]
    )
    assert code == 0
    data = json.loads(output.read_text(encoding="utf-8"))
    assert data["campaign_id"] == "demo"
    assert data["selection"]["batch_size"] == 8
    # Without a threshold the CLI must say that hit rates are unavailable.
    assert "success threshold" in capsys.readouterr().out


def test_init_campaign_from_fasta(tmp_path, reference):
    fasta = tmp_path / "ref.fasta"
    fasta.write_text(f">parent enzyme\n{reference}\n", encoding="utf-8")
    output = tmp_path / "campaign.json"
    code = main(
        [
            "init-campaign",
            "--id",
            "from-fasta",
            "--reference-fasta",
            str(fasta),
            "--assay-name",
            "activity",
            "--units",
            "AU",
            "-o",
            str(output),
        ]
    )
    assert code == 0
    assert json.loads(output.read_text(encoding="utf-8"))["reference_sequence"] == reference


def test_validate_reports_clean_inputs(inputs, capsys):
    code = main(
        [
            "validate",
            "--campaign",
            inputs["campaign"],
            "--measurements",
            inputs["measurements"],
            "--candidates",
            inputs["candidates"],
        ]
    )
    out = capsys.readouterr().out
    assert code == 0
    assert "No blocking problems found." in out
    assert "Evaluation protocol that would be used" in out


def test_validate_fails_on_blocking_rows(tmp_path, inputs, capsys):
    broken = pd.read_csv(inputs["measurements"])
    broken.loc[0, "sequence"] = "XXXX"
    path = tmp_path / "broken.csv"
    broken.to_csv(path, index=False)

    code = main(["validate", "--campaign", inputs["campaign"], "--measurements", str(path)])
    out = capsys.readouterr().out
    assert code == 1
    assert "cannot be used as given" in out


def test_plan_writes_exports_and_records_a_run(tmp_path, inputs, capsys):
    output = tmp_path / "plan"
    store = tmp_path / "campaigns.db"
    code = main(
        [
            "plan",
            "--campaign",
            inputs["campaign"],
            "--measurements",
            inputs["measurements"],
            "--candidates",
            inputs["candidates"],
            "--batch-size",
            "6",
            "--representation",
            "physchem",
            "--bootstrap",
            "20",
            "--cache-dir",
            str(tmp_path / "cache"),
            "-o",
            str(output),
            "--store",
            str(store),
        ]
    )
    out = capsys.readouterr().out
    assert code == 0
    assert "candidates selected" in out
    assert (output / "selected_variants.csv").exists()
    assert (output / "report.html").exists()
    assert store.exists()

    selected = pd.read_csv(output / "selected_variants.csv")
    assert len(selected) == 6

    from assaypilot import CampaignStore

    with CampaignStore(store) as opened:
        assert len(opened.list_runs()) == 1


def test_plan_refuses_broken_input_unless_asked(tmp_path, inputs, capsys):
    broken = pd.read_csv(inputs["candidates"])
    broken.loc[0, "sequence"] = "ZZZ"
    path = tmp_path / "broken_candidates.csv"
    broken.to_csv(path, index=False)

    arguments = [
        "plan",
        "--campaign",
        inputs["campaign"],
        "--measurements",
        inputs["measurements"],
        "--candidates",
        str(path),
        "--batch-size",
        "4",
        "--representation",
        "physchem",
        "--bootstrap",
        "10",
        "--cache-dir",
        str(tmp_path / "cache"),
        "-o",
        str(tmp_path / "plan"),
    ]
    assert main(arguments) == 1
    assert "cannot be used as given" in capsys.readouterr().err

    assert main(arguments + ["--skip-invalid"]) == 0


def test_evaluate_prints_a_recommendation(tmp_path, inputs, capsys):
    out_path = tmp_path / "evaluation.json"
    code = main(
        [
            "evaluate",
            "--campaign",
            inputs["campaign"],
            "--measurements",
            inputs["measurements"],
            "--candidates",
            inputs["candidates"],
            "--representation",
            "physchem",
            "--bootstrap",
            "20",
            "--cache-dir",
            str(tmp_path / "cache"),
            "-o",
            str(out_path),
        ]
    )
    out = capsys.readouterr().out
    assert code == 0
    assert "Evaluation protocol" in out
    assert "Recommendation" in out
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["recommendation"]["method"]


def test_demo_runs_end_to_end(tmp_path, capsys):
    code = main(
        [
            "demo",
            "-o",
            str(tmp_path / "demo"),
            "--n-measured",
            "60",
            "--n-candidates",
            "60",
            "--rank-variability",
            "0",
        ]
    )
    out = capsys.readouterr().out
    assert code == 0
    assert "synthetic data" in out
    assert (tmp_path / "demo" / "plan" / "report.html").exists()
    assert (tmp_path / "demo" / "data" / "README.md").exists()


def test_score_round_scores_a_previous_plan(tmp_path, inputs, capsys):
    output = tmp_path / "plan"
    main(
        [
            "plan",
            "--campaign",
            inputs["campaign"],
            "--measurements",
            inputs["measurements"],
            "--candidates",
            inputs["candidates"],
            "--batch-size",
            "6",
            "--representation",
            "physchem",
            "--bootstrap",
            "10",
            "--cache-dir",
            str(tmp_path / "cache"),
            "-o",
            str(output),
        ]
    )
    capsys.readouterr()

    selected = pd.read_csv(output / "selected_variants.csv")
    results = pd.DataFrame(
        {
            "variant_id": selected["design_id"],
            "sequence": selected["sequence"],
            "value": selected["score"] * 0.9,
            "condition_id": "37C_pH7.4",
            "round_id": "R4",
        }
    )
    results_path = tmp_path / "round4.csv"
    results.to_csv(results_path, index=False)

    card_path = tmp_path / "scorecard.json"
    code = main(
        [
            "score-round",
            "--manifest",
            str(output / "run_manifest.json"),
            "--measurements",
            str(results_path),
            "-o",
            str(card_path),
        ]
    )
    out = capsys.readouterr().out
    assert code == 0
    assert "Scorecard for" in out
    payload = json.loads(card_path.read_text(encoding="utf-8"))
    assert payload["n_matched"] == 6


def test_unknown_command_exits_with_usage_error():
    with pytest.raises(SystemExit) as excinfo:
        main(["not-a-command"])
    assert excinfo.value.code == 2


def test_a_missing_campaign_file_is_an_ordinary_error(tmp_path):
    """Not an AssayPilotError, but it must still not crash the process silently."""
    missing = tmp_path / "missing.json"
    with pytest.raises(FileNotFoundError):
        main(["validate", "--campaign", str(missing)])


def test_input_errors_are_reported_without_a_traceback(tmp_path, inputs, capsys):
    empty = tmp_path / "empty.csv"
    empty.write_text("variant_id,sequence,value,condition_id\n", encoding="utf-8")
    code = main(["validate", "--campaign", inputs["campaign"], "--measurements", str(empty)])
    assert code == 2
    assert "error:" in capsys.readouterr().err


def test_app_without_streamlit_explains_how_to_install(monkeypatch, capsys):
    """A missing optional extra must say what to install, not leak an import error."""
    import importlib.util

    real_find_spec = importlib.util.find_spec

    def missing_streamlit(name, *args, **kwargs):
        return None if name == "streamlit" else real_find_spec(name, *args, **kwargs)

    monkeypatch.setattr(importlib.util, "find_spec", missing_streamlit)
    assert main(["app"]) == 1
    error = capsys.readouterr().err
    assert 'pip install "assaypilot[app]"' in error
    assert "command line" in error
