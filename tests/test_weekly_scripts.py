import os
from pathlib import Path
import re
import json
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
WEEKS = ROOT / "weeks"


def test_there_is_one_script_for_each_roadmap_week() -> None:
    weeks = sorted(
        int(match.group(1))
        for script in WEEKS.glob("week_*/week_*.py")
        if (match := re.match(r"week_(\d{2})_", script.name))
    )
    assert weeks == list(range(1, 19))


def test_every_weekly_script_runs_and_saves_an_artifact(tmp_path: Path) -> None:
    environment = os.environ.copy()
    environment["COMPUTATIONAL_NEUROSCIENCE_OUTPUT_DIR"] = str(tmp_path)

    for script in sorted(WEEKS.glob("week_*/week_*.py")):
        if not re.match(r"week_\d{2}_", script.name):
            continue
        result = subprocess.run(
            [sys.executable, str(script)],
            cwd=ROOT,
            env=environment,
            check=True,
            capture_output=True,
            text=True,
            timeout=90,
        )
        week = int(script.name[5:7])
        artifacts = list((tmp_path / f"week_{week:02d}").iterdir())
        assert artifacts, f"{script.name} did not save any artifacts"
        assert "Saved:" in result.stdout, f"{script.name} did not report its output"


def test_weekly_runner_executes_a_script(tmp_path: Path) -> None:
    environment = os.environ.copy()
    environment["COMPUTATIONAL_NEUROSCIENCE_OUTPUT_DIR"] = str(tmp_path)
    result = subprocess.run(
        [sys.executable, str(WEEKS / "run_weekly.py"), "1"],
        cwd=ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert "Saved:" in result.stdout
    assert list((tmp_path / "week_01").iterdir())


def test_notebooks_match_their_source_scripts() -> None:
    sys.path.insert(0, str(ROOT / "tools"))
    from convert_to_notebooks import build_notebook

    # Extension notebooks are checked for sync too, but never executed here (they need their own environment).
    scripts = sorted(WEEKS.glob("week_*/week_*.py")) + sorted((ROOT / "extensions").glob("*/*.py"))
    for script in scripts:
        notebook = script.with_suffix(".ipynb")
        assert notebook.exists(), f"missing {notebook.name} next to {script.name}"
        expected = build_notebook(script.read_text(encoding="utf-8"))["cells"]
        actual = json.loads(notebook.read_text(encoding="utf-8"))["cells"]
        source = lambda cell: "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]
        assert [(c["cell_type"], source(c)) for c in actual] == [(c["cell_type"], c["source"]) for c in expected], (
            f"{notebook.name} is out of date: run python tools/convert_to_notebooks.py --execute"
        )
