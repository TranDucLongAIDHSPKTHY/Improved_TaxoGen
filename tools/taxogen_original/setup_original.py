"""Set up the original TaxoGen code (github.com/franticnerd/taxogen) as used by
the `taxogen_original` method: clone at the fixed commit, apply
original_code.patch (Python 3 syntax only, see README.md), compile word2vec.c
and create its Python 3.7 environment.

    python tools/taxogen_original/setup_original.py --python37 <path to python 3.7>
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_URL = "https://github.com/franticnerd/taxogen.git"
COMMIT = "929a7534fac89ad70b1920a962ef68e3d85888f3"
# word2vec.c calls posix_memalign, which Windows lacks; malloc only changes the alignment
CFLAGS = ["-O2", "-funroll-loops", "-Wno-unused-result", "-Dposix_memalign(p,a,s)=((*(p)=malloc(s))==NULL)"]


def sh(*cmd, cwd=None) -> None:
    print("[setup]", " ".join(map(str, cmd)))
    subprocess.run([str(c) for c in cmd], cwd=cwd, check=True)


def compile_word2vec(code: Path) -> None:
    exe = code / ("word2vec.exe" if sys.platform == "win32" else "word2vec")
    if shutil.which("gcc"):
        cc = ["gcc"]
    else:                               # pip install ziglang: a self-contained C compiler
        cc = [sys.executable, "-m", "ziglang", "cc"] + \
             (["-target", "x86_64-windows-gnu"] if sys.platform == "win32" else [])
    sh(*cc, *CFLAGS, "word2vec.c", "-o", exe.name, "-lm", cwd=code)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dest", type=Path, default=Path("external/taxogen_original"))
    ap.add_argument("--python37", required=True, help="a Python 3.7 interpreter")
    args = ap.parse_args()
    repo = args.dest / "repo"
    if not repo.exists():
        sh("git", "clone", REPO_URL, repo)
    sh("git", "checkout", "--quiet", COMMIT, cwd=repo)
    if subprocess.run(["git", "diff", "--quiet"], cwd=repo).returncode == 0:
        sh("git", "apply", "--whitespace=nowarn", HERE / "original_code.patch", cwd=repo)
    compile_word2vec(repo / "code")
    venv = args.dest / "venv37"
    if not venv.exists():
        sh(args.python37, "-m", "venv", venv)
    py = venv / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
    sh(py, "-m", "pip", "install", "-q", "-r", HERE / "requirements37.txt")
    print(f"[setup] done: code {repo / 'code'}, python {py}")


if __name__ == "__main__":
    main()
