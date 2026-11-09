# -*- coding: utf-8 -*-
"""Offline guard for the release builder itself: what ships must be complete, and rebuilding must
not destroy the tree it rebuilds into.

Two bugs this locks shut, both found while preparing the published tree:

1. A launcher mode that names a module the allow-list excludes. `run_route_b.ps1 -Mode calib`
   dispatched `knowledge_agents.calib_dump`, which did not ship, so a documented reproduction
   command raised ModuleNotFoundError on a fresh clone. The same applies to every
   `python -m knowledge_agents.*` the design document tells a reader to run.

2. Rebuilding into a directory that is also a git working copy. The old wipe called
   `shutil.rmtree(out)`, git writes its objects read-only, and the abort deleted HEAD, config and
   index while leaving objects behind -- an ordinary rebuild destroyed the repository.

Run:  .venv/Scripts/python.exe tools/test_release_integrity.py
"""
import importlib.util
import re
import stat
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

_spec = importlib.util.spec_from_file_location("package_release", ROOT / "tools" / "package_release.py")
pr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pr)

SHIPPED = {str(p.relative_to(pr.ROOT)).replace("\\", "/") for p in pr.collect()}

ok = True


def check(name, cond, detail=""):
    global ok
    ok = ok and bool(cond)
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))


def module_is_shipped(dotted: str) -> bool:
    """`knowledge_agents.a.b` resolves to a shipped file, not to a working-tree-only module."""
    rel = dotted.replace(".", "/")
    return f"{rel}.py" in SHIPPED or f"{rel}/__init__.py" in SHIPPED


# ---- 1. every entry point the shipped launcher and the documented commands name does ship -------
MODE_TARGET = re.compile(r"-m[\"'],?\s*[\"'](knowledge_agents[\w.]*)")
DOC_TARGET = re.compile(r"-m\s+[\"']?(knowledge_agents[\w.]*)")

targets = set()
launcher = (ROOT / "run_route_b.ps1").read_text(encoding="utf-8")
targets.update(MODE_TARGET.findall(launcher))
for doc in (ROOT / "knowledge_agents" / "ARCHITECTURE.md", ROOT / "README.md"):
    targets.update(DOC_TARGET.findall(doc.read_text(encoding="utf-8")))

unshipped = sorted(t for t in targets if not module_is_shipped(t))
check(f"all {len(targets)} documented module entry points ship", not unshipped,
      f"missing: {unshipped}" if unshipped else "(launcher modes + reproduction commands)")

# a documented -Mode with no dispatch branch would silently fall through to the demo path
default_mode = re.search(r'\[string\]\$Mode\s*=\s*"(\w+)"', launcher).group(1)
handled = set(re.findall(r'\$Mode\s+-eq\s+"?(\w+)"?', launcher)) | {default_mode}
documented = set(re.findall(r'-Mode\s+"?(\w+)"?', launcher))
for doc in (ROOT / "knowledge_agents" / "ARCHITECTURE.md", ROOT / "README.md"):
    documented |= set(re.findall(r'-Mode\s+"?(\w+)"?', doc.read_text(encoding="utf-8")))
stray = sorted(documented - handled)
check(f"all {len(documented)} documented -Mode values dispatch", not stray,
      f"no branch for: {stray}" if stray else f"handled: {sorted(handled)}")

# ---- 2. the tools/ allow-list and the frozen-table allow-list name real files ------------------
for name in pr.TOOLS_ALLOW:
    if f"tools/{name}" not in SHIPPED:
        check(f"tools allow-list entry ships: {name}", False, "not collected")
        break
else:
    check(f"all {len(pr.TOOLS_ALLOW)} tools/ allow-list entries ship", True)

# ---- 3. rebuild preserves version control and still clears read-only leftovers -----------------
tmp = Path(tempfile.mkdtemp(prefix="slc_release_wipe_"))
try:
    (tmp / "data").mkdir()
    (tmp / "data" / "stale.json").write_text("{}", encoding="utf-8")
    (tmp / ".git" / "objects" / "00").mkdir(parents=True)
    obj = tmp / ".git" / "objects" / "00" / "ab12cd34"
    obj.write_text("pack-contents", encoding="utf-8")
    obj.chmod(stat.S_IREAD)                       # exactly how git stores loose objects
    left = tmp / "leftover.txt"
    left.write_text("previous build", encoding="utf-8")
    left.chmod(stat.S_IREAD)                      # a read-only file outside .git must still go

    pr.wipe_previous_build(tmp)

    check("rebuild keeps .git objects in place", obj.is_file(),
          "a rebuild must never delete the caller's repository")
    check("rebuild removes the previous released content", not (tmp / "data").exists())
    check("rebuild deletes a read-only leftover", not left.exists())
finally:
    pr._rm_robust(tmp)

print("\n==== " + ("ALL PASS" if ok else "SOME FAILED") + " ====")
raise SystemExit(0 if ok else 1)
