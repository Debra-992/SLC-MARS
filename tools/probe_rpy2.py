"""Environment probe: rpy2 -> R -> DESeq2 / TCGAbiolinks / SummarizedExperiment.

Not a unit test. Run with the interpreter that has rpy2 installed (the tool venv,
``outputs\\mcp-venv``), after dot-sourcing the frozen environment so R_HOME / R_LIBS_USER
are correct for this machine:

    . tools\\slc_mars_env.ps1
    .\outputs\mcp-venv\Scripts\python.exe tools\probe_rpy2.py
"""
import os
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]

# Fall back to this project's own R library when the env script was not dot-sourced.
os.environ.setdefault("R_HOME", r"C:\Program Files\R\R-4.6.1")
os.environ.setdefault("R_LIBS_USER", str(ROOT / "outputs" / "R-libs"))
for sub in (("bin", "x64"), ("bin",)):
    p = os.path.join(os.environ["R_HOME"], *sub)
    if os.path.isdir(p):
        os.add_dll_directory(p)

try:
    import rpy2.robjects as ro
    from rpy2.robjects.packages import importr
except ModuleNotFoundError as e:
    sys.exit(f"rpy2 unavailable in this interpreter ({e.name}). "
             f"Use outputs\\mcp-venv\\Scripts\\python.exe, not .venv.")

d = importr("DESeq2")
t = importr("TCGAbiolinks")
se = importr("SummarizedExperiment")


def ver(pkg):
    return str(ro.r(f"as.character(packageVersion('{pkg}'))")[0])


print("rpy2 loaded OK")
print("R_LIBS_USER :", os.environ["R_LIBS_USER"])
print("DESeq2      :", d.__rname__, ver("DESeq2"))
print("TCGAbiolinks:", t.__rname__, ver("TCGAbiolinks"))
print("SummExp     :", se.__rname__, ver("SummarizedExperiment"))
