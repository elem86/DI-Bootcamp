"""Repository checks: documented commands, links, dependencies and tracked files.

None of these need the data, the saved models or a network connection, so
they run in a fresh clone and in continuous integration.
"""

import ast
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT = Path(__file__).resolve().parents[1]
DOCUMENTS = [PROJECT / "README.md", *sorted((PROJECT / "docs").glob("*.md"))]
NOTEBOOKS = sorted((PROJECT / "notebooks").glob("*.ipynb"))

COMMAND = re.compile(r"python -m src\.(\w+)")
LINK = re.compile(r"\]\(([^)\s]+)\)")
HEADING = re.compile(r"^#{1,6} +(.+?) *$", re.MULTILINE)
CODE_BLOCK = re.compile(r"^```.*?^```", re.MULTILINE | re.DOTALL)

# Import name -> distribution name, where they differ
DISTRIBUTIONS = {"sklearn": "scikit-learn"}
LOCAL_PACKAGES = {"src", "tests"}

MAX_TRACKED_BYTES = 3 * 1024 * 1024  # the largest legitimate file is an executed notebook of about 1.4 MB
# Absolute paths from a developer's machine (split so this file does not match itself)
LOCAL_PATH = re.compile(r"[A-Za-z]:\\" + r"(Users|Documents and Settings)\\|/" + r"(Users|home)/[\w.-]+/")
TEXT_SUFFIXES = {".py", ".md", ".txt", ".toml", ".ipynb", ".yml", ".yaml", ".cfg", ".ini", ".json", ""}


# ---------------------------------------------------------------- documentation


def anchor(heading: str) -> str:
    """GitHub's anchor for a Markdown heading."""
    text = re.sub(r"[`*]", "", heading).strip().lower()
    return re.sub(r"[^\w\- ]", "", text).replace(" ", "-")


def anchors(path: Path) -> set[str]:
    text = CODE_BLOCK.sub("", path.read_text(encoding="utf-8"))
    return {anchor(h) for h in HEADING.findall(text)}


def notebook_text(path: Path) -> str:
    cells = json.loads(path.read_text(encoding="utf-8"))["cells"]
    return "\n".join("".join(cell["source"]) for cell in cells)


def test_documented_commands_exist():
    """Every `python -m src.<module>` in the docs and notebooks names a runnable module."""
    sources = {path.name: path.read_text(encoding="utf-8") for path in DOCUMENTS}
    sources |= {path.name: notebook_text(path) for path in NOTEBOOKS}
    commands = {(name, module) for name, text in sources.items() for module in COMMAND.findall(text)}
    assert commands, "no commands found in the documentation"
    for name, module in sorted(commands):
        path = PROJECT / "src" / f"{module}.py"
        assert path.exists(), f"{name} documents `python -m src.{module}`, but src/{module}.py does not exist"
        assert 'if __name__ == "__main__":' in path.read_text(encoding="utf-8"), \
            f"{name} documents `python -m src.{module}`, but it has no command-line entry point"


@pytest.mark.parametrize("document", DOCUMENTS, ids=lambda path: path.name)
def test_documentation_links_resolve(document):
    """Relative links and images point at files in the repository, and #anchors at real headings."""
    text = CODE_BLOCK.sub("", document.read_text(encoding="utf-8"))
    for target in LINK.findall(text):
        if re.match(r"[a-z]+:", target):  # https:, mailto:
            continue
        file_part, _, fragment = target.partition("#")
        linked = (document.parent / file_part).resolve() if file_part else document
        assert linked.exists(), f"{document.name} links to missing {target}"
        if fragment and linked.suffix == ".md":
            assert fragment in anchors(linked), f"{document.name} links to missing heading {target}"


FIGURE_NAME = re.compile(r"(?:outputs/figures|docs/screenshots)/[\w.-]+\.png")


def test_documented_figure_files_exist():
    """Figures and screenshots named in the docs (also outside links, e.g. in the presentation outline) exist."""
    names = {name for path in DOCUMENTS for name in FIGURE_NAME.findall(path.read_text(encoding="utf-8"))}
    assert names, "no figure names found in the documentation"
    missing = sorted(name for name in names if not (PROJECT / name).exists())
    assert not missing, f"documented figures that do not exist: {missing}"


# ---------------------------------------------------------------- dependencies


def requirement_names(path: Path) -> set[str]:
    names = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.split("#")[0].strip()
        if line and not line.startswith("-"):
            names.add(re.split(r"[<>=!~\[ ;]", line)[0].lower())
    return names


def imported_packages() -> set[str]:
    sources = [p.read_text(encoding="utf-8") for p in [PROJECT / "app.py", *(PROJECT / "src").glob("*.py"),
                                                         *(PROJECT / "tests").glob("*.py")]]
    for path in NOTEBOOKS:
        for cell in json.loads(path.read_text(encoding="utf-8"))["cells"]:
            if cell["cell_type"] == "code":
                lines = "".join(cell["source"]).splitlines()
                sources.append("\n".join(line for line in lines if not line.lstrip().startswith(("%", "!"))))
    packages = set()
    for source in sources:
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.Import):
                packages |= {alias.name.split(".")[0] for alias in node.names}
            elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                packages.add(node.module.split(".")[0])
    return packages - set(sys.stdlib_module_names) - LOCAL_PACKAGES - {"__future__"}


def test_every_imported_package_is_a_declared_requirement():
    declared = requirement_names(PROJECT / "requirements.txt")
    needed = {DISTRIBUTIONS.get(name, name).lower() for name in imported_packages()}
    assert needed <= declared, f"imported but not in requirements.txt: {sorted(needed - declared)}"


def test_python_version_is_consistent():
    """pyproject.toml, the README, the lock file and CI all name the one tested Python version."""
    import tomllib

    requires = tomllib.loads((PROJECT / "pyproject.toml").read_text(encoding="utf-8"))["project"]["requires-python"]
    match = re.fullmatch(r">=(\d+)\.(\d+)", requires)
    assert match, f"unexpected requires-python: {requires}"
    major, minor = map(int, match.groups())
    version = f"{major}.{minor}"
    assert sys.version_info[:2] >= (major, minor)
    assert f"Python {version}" in (PROJECT / "README.md").read_text(encoding="utf-8")
    assert f"Python {version}." in (PROJECT / "requirements-lock.txt").read_text(encoding="utf-8")
    # The workflow sits at the root of the DI-Bootcamp repository (see the README); skip it if this is not that layout
    workflow = PROJECT.parent / ".github" / "workflows" / "football-match-predictor.yml"
    if workflow.exists():
        assert f'python-version: "{version}"' in workflow.read_text(encoding="utf-8")


def test_lock_file_pins_the_same_packages():
    lock = PROJECT / "requirements-lock.txt"
    assert requirement_names(lock) == requirement_names(PROJECT / "requirements.txt")
    for line in lock.read_text(encoding="utf-8").splitlines():
        line = line.split("#")[0].strip()
        if line and not line.startswith("-"):
            assert re.fullmatch(r"[\w.-]+==[\w.]+", line), f"not an exact pin: {line}"


# ---------------------------------------------------------------- tracked files


@pytest.fixture(scope="module")
def tracked_files() -> list[str]:
    git = shutil.which("git")
    if git is None:
        pytest.skip("git is not installed")
    result = subprocess.run([git, "ls-files", "-z"], cwd=PROJECT, capture_output=True, text=True)
    if result.returncode != 0:
        pytest.skip("not a git checkout")
    files = [name for name in result.stdout.split("\0") if name]
    if not files:
        pytest.skip("no tracked files")
    return files


def test_no_data_models_or_secrets_are_tracked(tracked_files):
    """Downloaded data, generated results, models and secrets stay out of git (only placeholders are tracked)."""
    for name in tracked_files:
        path = Path(name)
        if path.parts[0] == "data":
            assert path.name == ".gitkeep", f"data file is tracked: {name}"
        if path.parts[0] == "outputs" and path.parts[1] != "figures":
            assert path.name == ".gitkeep", f"generated output is tracked: {name}"
        assert path.suffix not in {".joblib", ".pkl", ".pickle", ".xlsx", ".parquet"}, f"binary artifact: {name}"
        assert path.name not in {".env", "secrets.toml"} and not path.name.startswith(".env."), f"secret: {name}"


def test_tracked_files_are_small(tracked_files):
    for name in tracked_files:
        path = PROJECT / name
        if path.exists():
            assert path.stat().st_size <= MAX_TRACKED_BYTES, f"{name} is {path.stat().st_size / 1e6:.1f} MB"


def test_no_local_machine_paths_are_tracked(tracked_files):
    this_file = Path(__file__).resolve()
    for name in tracked_files:
        path = PROJECT / name
        if path.suffix not in TEXT_SUFFIXES or not path.exists() or path.resolve() == this_file:
            continue
        match = LOCAL_PATH.search(path.read_text(encoding="utf-8", errors="ignore"))
        assert match is None, f"{name} contains a local path: {match.group(0) if match else ''}"
