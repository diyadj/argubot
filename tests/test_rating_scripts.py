"""The rating sheet script must never write over sheets that may already hold someone's ratings."""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _transcripts(folder: Path) -> None:
    folder.mkdir(parents=True)
    for system in ("B", "C1"):
        rows = [
            {"dialogue_id": f"d{i}", "system": system, "turns": [
                {"user_text": "Uniforms should go.", "bot_text": f"{system} says no ({i}).", "claims_in_prompt": [],
                 "passages_in_prompt": [], "candidates": [], "prompt": ""}]}
            for i in range(3)
        ]
        (folder / f"{system}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))


def _make(tmp_path, *extra):
    return subprocess.run(
        [sys.executable, "scripts/make_rating_sheets.py", "--transcripts", str(tmp_path / "tr"),
         "--out", str(tmp_path / "sheets"), "--systems", "B", "C1", *extra],
        cwd=ROOT, capture_output=True, text=True,
    )


def test_sheets_are_written_once_and_never_overwritten_by_accident(tmp_path):
    _transcripts(tmp_path / "tr")
    first = _make(tmp_path)
    assert first.returncode == 0, first.stderr
    sheet = tmp_path / "sheets" / "rater_1.csv"
    sheet.write_text(sheet.read_text() + "pretend someone rated this\n")

    again = _make(tmp_path)
    assert again.returncode != 0 and "may hold ratings" in (again.stdout + again.stderr)
    assert sheet.read_text().endswith("pretend someone rated this\n")

    forced = _make(tmp_path, "--overwrite")
    assert forced.returncode == 0
    assert "pretend someone rated this" not in sheet.read_text()
    assert (tmp_path / "sheets" / "key_PRIVATE.csv").exists()


def test_precision_sheet_saved_as_csv_utf8_scores_from_any_folder(tmp_path):
    # Excel and Google Sheets save "CSV UTF-8" with a byte-order mark; the scorer must still read it.
    sheets, results = tmp_path / "precision", tmp_path / "results"
    sheets.mkdir()
    results.mkdir()
    (sheets / "precision_key_PRIVATE.csv").write_text(
        "pair_id,system,dialogue_id,turn,score\npair000,B,d0,1,0.9\npair001,human,d1,2,0.8\n", encoding="utf-8")
    (sheets / "precision_sheet.csv").write_text(
        "pair_id,earlier_claim,new_sentence,real_contradiction,notes\npair000,a,b,1,\npair001,c,d,0,\n",
        encoding="utf-8-sig")
    (results / "metrics_strict.json").write_text(json.dumps(
        {"B": {"contradiction_rate": 0.1}, "human": {"contradiction_rate": 0.05}}))

    run = subprocess.run(
        [sys.executable, "scripts/make_precision_sheet.py", "--score", "--dir", str(sheets), "--results", str(results)],
        cwd=ROOT, capture_output=True, text=True,
    )
    assert run.returncode == 0, run.stderr
    summary = json.loads((results / "precision_strict.json").read_text())
    assert summary["B"]["precision"] == 1.0 and summary["human"]["precision"] == 0.0
