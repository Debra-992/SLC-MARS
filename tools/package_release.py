# -*- coding: utf-8 -*-
"""Build the public release tree for SLC-MARS.

Why not simply ``git init`` inside the working directory:
  * the working tree carries ~1.6 GB of virtualenvs and ~1.5 GB of regenerable caches
    (R libraries, three baseline environments, DESeq2/GDC and KM-TIMER caches). Those
    neither push nor belong in a public repository;
  * the release tree is a build product, so machine-local paths inside it may be
    rewritten to portable placeholders without touching the working copy.

What ships (allow-list only -- anything not listed is excluded):
  in   knowledge_agents/  tools/ (named scripts)  docs/  data/  evaluation/
       outputs/traces/ (frozen result tables, public inputs, ground truth, fitted models)
       plus the root configuration files and README.md
  out  paper/ (published separately), MetaGPT/ (third-party framework with its own
       licence; README gives the install route), .venv/, .cache/, logs/,
       secrets.local.ps1, and every regenerable cache under outputs/

Published numbers come from frozen runs, so raw per-question traces are not shipped.
Result tables and fitted model artifacts are; the release reproduces behaviour, not the
exact token counts of the original runs.

Build fails on any of: a machine-local absolute path, a residual CJK character, a
version-arm token, a conversation/author-confirmation residue, a non-ASCII filename, a
scratch-style filename, or a file over the GitHub single-file limit.

Run:  .venv/Scripts/python.exe tools/package_release.py [--dry-run] [--out DIR] [--force]
"""
from __future__ import annotations

import argparse
import hashlib
import os
import re
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / ".release" / "SLC-MARS"
MARKER = ".slc_release_build"
PACKAGE_NAME = "SLC-MARS"
# directories that survive a rebuild because they belong to the user's version control, not to us
VCS_DIRS = {".git", ".hg", ".svn"}

INCLUDE_FILES = [
    ".gitignore",
    "LICENSE",
    "README.md",
    "config2.yaml",
    "run_route_b.ps1",
    "secrets.example.ps1",
]

# knowledge_agents modules that are experiment scaffolding rather than part of the released
# pipeline. The rule is mechanical: anything the shipped launcher can dispatch (a run_route_b.ps1
# mode) or that a documented reproduction command names must ship, because a command that raises
# ModuleNotFoundError on a fresh clone is a release bug. What stays here has no entry point in the
# released tree and only reads or writes working-tree fitting inputs.
# The conformal risk-control algorithm itself (conformal.py) ships, because the decision layer's
# SLC_CONFORMAL_FILE hook reads what it produces.
AGENT_EXCLUDE = {
    "feature_vector.py",
    "probe_all_tools.py",
}

# Directories taken wholesale (subject to SKIP_* below).
INCLUDE_DIRS = ["knowledge_agents", "docs", "data", "evaluation"]

# tools/ is opt-in: experiment drivers and one-off launchers stay in the working tree.
TOOLS_ALLOW = [
    "clean_questions.py",
    "gen_public_inputs.py",
    "label_antimemorization.py",
    "load_secrets.ps1",
    "package_release.py",
    "run_baselines_all500.ps1",
    "slc_mars_env.ps1",
    "sync_metagpt_config.ps1",
    "test_answer_ensemble.py",
    "test_direction_logic.py",
    "test_llm_chief.py",
    "test_llm_settings_env.py",
    "test_negative_evidence.py",
    "test_no_secrets.py",
    "test_pairwise_graph_model.py",
    "test_paralog_conformal.py",
    "test_prior_ensemble.py",
    "test_prior_retrieve.py",
    "test_release_integrity.py",
    "test_triad_critique.py",
    "test_triad_vote.py",
]

# outputs/traces/: frozen result tables, shared inputs/ground truth and fitted models.
TRACES_ALLOW = [
    "antimemorization_labels_all489.json",
    "baseline_final_report.txt",
    "baselines_by_dataset_report.txt",
    "baselines_gt.json",
    "baselines_input.json",
    "baselines_table_master.md",
    "compare_table_n489_seed5.md",
    "conformal_report.md",
    "conformal_thresholds.json",
    "dataset_b_qids_all489.json",
    "de_panel_ext.json",
    "efficiency_accuracy_master.txt",
    "fullfeature_model.json",
    "manifest_all489.json",
    "pairwise_graph_model.json",
    "pipeline_full489_report.txt",
    "sample_manifest.json",
    "topk_eval_summary.txt",
]

# outputs/examples/: a couple of translated end-to-end traces so the message shapes are
# visible without shipping the 48 MB cache.
EXAMPLES_DIR = "outputs/examples"

SKIP_DIR_NAMES = {"__pycache__", ".ipynb_checkpoints", "node_modules", ".vscode", "logs", ".cache"}
SKIP_FILE_NAMES = {"secrets.local.ps1"}
SKIP_SUFFIX = {".pyc", ".pyo", ".log", ".err", ".tmp", ".bak", ".aux", ".fls",
               ".fdb_latexmk", ".synctex.gz", ".local.ps1"}
MAX_FILE_BYTES = 90 * 1024 * 1024          # GitHub hard-limits a file at 100 MB; keep margin
TEXT_SUFFIX = {".json", ".jsonl", ".md", ".tex", ".py", ".ps1", ".yaml", ".yml",
               ".txt", ".csv", ".R", ".sh"}

# ---------------------------------------------------------------- gates

# CJK writing, and nothing else: ideographs (uniform, extension A, compatibility), kana, CJK
# punctuation and strokes, radicals, and the full-width forms.  English typography is deliberately
# outside the class -- em dashes, curly quotes, the Greek letters in gene symbols and the
# box-drawing of the layout diagram are all legitimate in an English document, and untranslated
# text always carries ideographs, so dropping the punctuation characters loses no detection power
# while stopping the gate from demanding ASCII prose.
CJK = re.compile("[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\u2e80-\u2eff\u3000-\u303f\u3040-\u30ff"
                 "\u31c0-\u31ef\u31f0-\u31ff\uff01-\uff5e]")

# Internal experiment-arm identifiers. These are iteration bookkeeping; the released
# system is SLC-MARS and names itself in exactly one way. The bracketed digit forms below
# do not match this file's own source, so no split-literal trick is needed here.
#
# A bare match on the word alone is not enough: journal volume citations such as a DOI
# "10.4239/wjd.v16.i8.104879" carry the same shape and must not fail the build, so the
# standalone form may not be preceded by a word character or a dot. Underscore/hyphen
# suffixed artifacts (answer_vNN.jsonl) are still caught by the second alternative, and
# scratch filenames of that shape by SCRATCH_NAME below.
VERSION_ARM = re.compile(r"(?<![\w.])\bV(?:1[0-9]|2[0-9]|500)\b|[-_]V(?:1[0-9]|2[0-9]|500)\b", re.I)

# Notes that record a decision as a dialogue event rather than as a design rationale.
# The Chinese phrases are written as escapes and the English ones are assembled from word
# lists, so that this file's own source never contains a full match (same trick as
# PERSONAL_MARKERS below). The Chinese entries are the seven ways of writing that a
# content decision was settled between the implementer and the reviewer; the English
# entries are the corresponding phrasings, plus references to chat/session transcripts.
RESIDUE_ZH = ("\\u4e0e\\u4f5c\\u8005\\u786e\\u8ba4", "\\u4e0e\\u7528\\u6237\\u786e\\u8ba4",
              "\\u4f5c\\u8005\\u786e\\u8ba4", "\\u7528\\u6237\\u786e\\u8ba4",
              "\\u6309\\u4f60\\u7684\\u8981\\u6c42", "\\u5982\\u4f60\\u6240",
              "\\u4f60\\u6211\\u4e4b\\u95f4")
RESIDUE_VERBS = "confirmed|agreed|discussed|requested|specified|asked|wants|said"
RESIDUE_PARTIES = "author|user"
RESIDUE_NOUNS = "chat|conversation|session|context"
RESIDUE_TAILS = "history|summary|compaction"
RESIDUE_EN = (
    rf"(?:{RESIDUE_VERBS})(?:ing)? (?:with |by )?(?:the )?(?:{RESIDUE_PARTIES})",
    rf"(?:per our|according to the) (?:{RESIDUE_PARTIES}|{RESIDUE_NOUNS})",
    rf"(?:{RESIDUE_NOUNS}) (?:{RESIDUE_TAILS})",
)
RESIDUE = re.compile("|".join([z for z in RESIDUE_ZH]
                              + [f"(?i:{p})" for p in RESIDUE_EN]))

# Machine-local paths that must never be published. The directory name is split in two
# literals on purpose: this file ships too, and one literal would match its own regex.
PERSONAL_MARKERS = ("xwechat" "_files", "wxshare" "files")
PERSONAL_PATH = re.compile(
    r"(?i)[a-z]:[\\/]+[^\s\"'<>]*(?:" + "|".join(re.escape(m) for m in PERSONAL_MARKERS) + r")[^\s\"'<>]*")

# Scratch naming: an underscore prefix (other than Python's dunder files), a version
# suffix, or a smoke-test marker means a working-tree file rather than a released one.
SCRATCH_NAME = re.compile(r"^_(?!_)|[-_]v(?:1[0-9]|2[0-9]|500)\b|_smoke", re.I)


def path_pattern(win_path: str) -> re.Pattern:
    """Compile a Windows absolute path into a regex insensitive to separator doubling.

    The same path appears as ``E:\\\\bundle\\\\...`` inside JSON, with single backslashes in
    plain text, and sometimes with forward slashes; ``[\\\\/]+`` consumes any mix.
    """
    segs = [s for s in re.split(r"[\\/]+", win_path.strip().strip('"').rstrip("\\")) if s]
    return re.compile("(?i)" + r"[\\/]{1,2}".join(re.escape(s) for s in segs))


def build_scrubs() -> list[tuple[re.Pattern, str]]:
    """Machine-local prefixes to replace. The handoff root is read from the environment
    and never hardcoded into this file."""
    scrubs: list[tuple[re.Pattern, str]] = [(path_pattern(str(ROOT)), "<PROJECT_ROOT>")]
    handoff = os.environ.get("SLC_ROOT", "").strip()
    if handoff:
        scrubs.insert(0, (path_pattern(handoff), "<SLC_ROOT>"))
    else:
        print("[warn] $env:SLC_ROOT is unset: the handoff root prefix is not rewritten;"
              " only the PERSONAL_PATH gate protects the tree"
              " (source tools/slc_mars_env.ps1 before packaging)")
    return scrubs


def collect() -> list[Path]:
    """Collect source files to ship, in the allow-listed order, relative to the repo root."""
    picked: list[Path] = []
    for name in INCLUDE_FILES:
        p = ROOT / name
        if not p.is_file():
            raise SystemExit(f"[fatal] missing root file {name}")
        picked.append(p)
    for d in INCLUDE_DIRS:
        base = ROOT / d
        if not base.is_dir():
            raise SystemExit(f"[fatal] missing directory {d}")
        for p in sorted(base.rglob("*")):
            rel = p.relative_to(ROOT)
            if p.is_dir():
                continue
            if any(part in SKIP_DIR_NAMES for part in rel.parts[:-1]):
                continue
            if rel.name in SKIP_FILE_NAMES or p.suffix.lower() in SKIP_SUFFIX:
                continue
            if rel.parts[0] == "knowledge_agents" and p.name in AGENT_EXCLUDE:
                continue
            picked.append(p)
    for name in TOOLS_ALLOW:
        p = ROOT / "tools" / name
        if not p.is_file():
            raise SystemExit(f"[fatal] tools allow-list names a missing file: tools/{name}")
        picked.append(p)
    for name in TRACES_ALLOW:
        p = ROOT / "outputs" / "traces" / name
        if not p.is_file():
            raise SystemExit(f"[fatal] outputs/traces allow-list names a missing file: {name}")
        picked.append(p)
    examples = ROOT / EXAMPLES_DIR
    if examples.is_dir():
        for p in sorted(examples.glob("*")):
            if p.is_file() and p.suffix.lower() in TEXT_SUFFIX:
                picked.append(p)
    else:
        print(f"[warn] {EXAMPLES_DIR}/ is absent: no example traces will ship")
    return picked


def emit_text(src: Path, dst: Path, scrubs, stats: dict, dry: bool) -> None:
    raw = src.read_bytes()
    bom = raw.startswith(b"\xef\xbb\xbf")
    try:
        text = raw.decode("utf-8-sig" if bom else "utf-8")
    except UnicodeDecodeError:                      # non-UTF-8 text: copy verbatim, still gated
        stats["copy_raw"] += 1
        latin = raw.decode("latin-1")
        if PERSONAL_PATH.search(latin):
            stats["personal_hits"].append(str(src.relative_to(ROOT)))
        stats["cjk_hits"].append(str(src.relative_to(ROOT)))
        if not dry:
            dst.write_bytes(raw)
        return
    for rx, token in scrubs:
        text, n = rx.subn(token, text)
        stats["rewrites"] += n
    rel = str(src.relative_to(ROOT)).replace("\\", "/")
    if PERSONAL_PATH.search(text):
        stats["personal_hits"].append(rel)
    if CJK.search(text):
        stats["cjk_hits"].append(rel)
    if VERSION_ARM.search(text):
        stats["version_hits"].append(rel)
    if RESIDUE.search(text):
        stats["residue_hits"].append(rel)
    if not dry:
        dst.write_bytes((("\ufeff" + text) if bom else text).encode("utf-8"))


def run_gates(stats: dict, files: list[Path]) -> int:
    """Fail the build on any policy violation. All findings are printed before the exit."""
    rc = 0

    def report(title: str, items: list[str], hint: str) -> None:
        nonlocal rc
        if not items:
            return
        rc = 1
        print(f"\n[fatal] {title} ({len(items)} file(s)):")
        for h in items[:25]:
            print("   -> " + h)
        if len(items) > 25:
            print(f"   ... and {len(items) - 25} more")
        print("   " + hint)

    report("machine-local absolute path", stats["personal_hits"],
           "these must already be rewritten to <PROJECT_ROOT> / <SLC_ROOT>")
    report("residual CJK characters", stats["cjk_hits"],
           "the public tree is English-only; translate or drop the text")
    report("internal version-arm token", stats["version_hits"],
           "the released system is called SLC-MARS; rename the arms")
    report("conversation / author-confirmation residue", stats["residue_hits"],
           "state a design rationale instead of recording who agreed with whom")

    bad_names = [str(p.relative_to(ROOT)).replace("\\", "/") for p in files
                 if any(ord(c) > 127 for c in p.name)]
    report("non-ASCII filename", bad_names, "rename the file in the working tree")

    scratch = [str(p.relative_to(ROOT)).replace("\\", "/") for p in files
               if SCRATCH_NAME.search(p.name) and p.name != MARKER]
    report("scratch/version-named filename", scratch,
           "intermediate-process files do not belong in the release")

    if stats["skipped_big"]:
        rc = 1
        print(f"\n[fatal] {stats['skipped_big']} file(s) exceed the GitHub size limit; "
              f"adjust the allow-list")
    return rc


def _rm_robust(path: Path) -> None:
    """Delete a file or subtree, clearing the read-only bit when Windows refuses.

    Version-control stores mark pack and loose objects read-only on purpose, so a plain rmtree aborts
    half-way through .git and leaves the repository without HEAD/config/index while the working files
    are still there. Retrying alone does not help: the refusal is the file attribute.
    """
    if path.is_dir() and not path.is_symlink():
        for child in sorted(path.iterdir()):
            _rm_robust(child)
        path.rmdir()
        return
    try:
        path.unlink()
    except PermissionError:
        os.chmod(path, 0o666)
        path.unlink()


def wipe_previous_build(out: Path) -> None:
    """Remove the previous build's released content, but never a version-control directory.

    The output tree routinely doubles as the repository that gets pushed: rebuilding into it must not
    destroy its history.
    """
    for entry in sorted(out.iterdir()):
        if entry.name in VCS_DIRS:
            continue
        _rm_robust(entry)


def build(out: Path, dry: bool, force: bool = False) -> int:
    scrubs = build_scrubs()
    files = collect()
    stats = {"copied": 0, "skipped_big": 0, "copy_raw": 0, "rewrites": 0,
             "personal_hits": [], "cjk_hits": [], "version_hits": [], "residue_hits": []}
    total = 0
    per_top: dict[str, list[int]] = {}
    largest = (0, "")

    if out.exists() and not dry:
        if not (out / MARKER).exists() and any(out.iterdir()) and not force:
            raise SystemExit(f"[fatal] {out} exists without this script's build marker; "
                             f"refusing to wipe it. Add --force if you are sure it is ours "
                             f"(an interrupted build leaves exactly this state)")
        # Windows indexers and antivirus tools briefly hold large jsonl files; back off and
        # retry instead of leaving a half-deleted tree behind.
        for attempt in range(6):
            try:
                wipe_previous_build(out)
                break
            except PermissionError:
                if attempt == 5:
                    raise
                time.sleep(1.5)

    for src in files:
        size = src.stat().st_size
        rel = src.relative_to(ROOT)
        if size > MAX_FILE_BYTES:
            stats["skipped_big"] += 1
            print(f"[skip >90MB] {rel} ({size / 1e6:.1f} MB)")
            continue
        top = rel.parts[0] if rel.parts[0] != "outputs" else "/".join(rel.parts[:2])
        per_top.setdefault(top, [0, 0])
        per_top[top][0] += 1
        per_top[top][1] += size
        total += size
        if size > largest[0]:
            largest = (size, str(rel))
        dst = out / rel
        # dry-run still sanitizes and passes the gates, otherwise the check finds nothing
        if not dry:
            dst.parent.mkdir(parents=True, exist_ok=True)
        if src.suffix.lower() in TEXT_SUFFIX:
            emit_text(src, dst, scrubs, stats, dry)
        elif not dry:
            shutil.copy2(src, dst)
        stats["copied"] += 1

    if not dry:
        out.mkdir(parents=True, exist_ok=True)
        (out / MARKER).write_text(f"built by tools/package_release.py for {PACKAGE_NAME}\n",
                                  encoding="utf-8")

    print(f"\n==== release package {'(dry-run)' if dry else str(out)} ====")
    for top in sorted(per_top):
        n, b = per_top[top]
        print(f"  {top:<24} {n:>6} files  {b / 1e6:>8.1f} MB")
    print(f"  {'TOTAL':<24} {stats['copied']:>6} files  {total / 1e6:>8.1f} MB")
    print(f"  largest={largest[1]} ({largest[0] / 1e6:.1f} MB)")
    print(f"  path rewrites={stats['rewrites']}  non-utf8 copies={stats['copy_raw']}  "
          f"over-limit skipped={stats['skipped_big']}")
    tbl = out / "outputs" / "traces" / "baselines_table_master.md"
    if not dry and tbl.is_file():
        print("  sha256 baselines_table_master.md = "
              + hashlib.sha256(tbl.read_bytes()).hexdigest()[:16])

    rc = run_gates(stats, files)
    if rc == 0:
        print("\n[ok] all release gates passed")
    return rc


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=str(DEFAULT_OUT), help="release tree output directory")
    ap.add_argument("--dry-run", action="store_true", help="compute and gate without writing")
    ap.add_argument("--force", action="store_true",
                    help="allow wiping an output directory that lacks the build marker")
    a = ap.parse_args()
    return build(Path(a.out).resolve(), a.dry_run, a.force)


if __name__ == "__main__":
    sys.exit(main())
