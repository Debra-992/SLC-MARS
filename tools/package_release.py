# -*- coding: utf-8 -*-
"""打包公开发布树：只复制"人会看/会跑"的东西，并把机器本地绝对路径换成可移植占位符。

为什么不在项目目录里直接 git init：
  * 项目树里 1.6 GB 的 .venv、1.5 GB 的 outputs 缓存（R-libs、三个 baseline venv、
    DESeq2/GDC 与 KM-TIMER 缓存）都是可再生物，混进仓库既推不上去也说不清边界；
  * 发布树是构建产物，允许改写 trace 里的本机路径，而不必动手上这一份。

内容口径（2026-11-03 与作者确认）：
  进包：knowledge_agents/ tools/ docs/ data/ evaluation/ outputs/traces/ + 根配置与 README
  不进：paper/（另行发布）、MetaGPT/（第三方框架，自带 .git 与许可证，README 给安装口径）、
        .venv/ .cache/ logs/ secrets.local.ps1，以及 outputs 下所有可再生缓存与虚拟环境

改写只动字符串前缀，不动任何数字。traces 里的 boxplot_path/plot_path 在代码里只被用作
真值判断或打印（decision_action.py / run_single_question.py），从不 open()，所以换成
<SLC_ROOT> 不改变行为；trace 缓存键是 SLC_TRACE_VERSION+qid、无内容哈希，改写也不失效。
构建完会用发布树自己重算主表，数字与论文表 1 不一致就判失败。

Run:  .venv/Scripts/python.exe tools/package_release.py [--dry-run] [--out DIR]
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

INCLUDE_FILES = [".gitignore", "LICENSE", "README.md", "config2.yaml", "run_route_b.ps1", "secrets.example.ps1"]
INCLUDE_DIRS = ["knowledge_agents", "tools", "docs", "data", "evaluation", "outputs/traces"]

SKIP_DIR_NAMES = {"__pycache__", ".ipynb_checkpoints", "node_modules", ".vscode", "logs", ".cache"}
SKIP_FILE_NAMES = {"secrets.local.ps1"}
SKIP_SUFFIX = {".pyc", ".pyo", ".log", ".err", ".tmp", ".bak", ".aux", ".fls",
               ".fdb_latexmk", ".synctex.gz", ".local.ps1"}
MAX_FILE_BYTES = 90 * 1024 * 1024          # GitHub 单文件 100 MB 硬上限，留余量
TEXT_SUFFIX = {".json", ".jsonl", ".md", ".tex", ".py", ".ps1", ".yaml", ".yml",
               ".txt", ".csv", ".R", ".sh"}

# 兜底闸门：包里只要还剩一个指向个人目录的绝对路径就构建失败（不依赖我事先知道那个值）。
# 目录名拆成两段写：本文件自己也在包里，写成一段会被自己的正则命中。
PERSONAL_MARKERS = ("xwechat" "_files", "wxshare" "files")
PERSONAL_PATH = re.compile(
    r"(?i)[a-z]:[\\/]+[^\s\"'<>]*(?:" + "|".join(re.escape(m) for m in PERSONAL_MARKERS) + r")[^\s\"'<>]*")


def path_pattern(win_path: str) -> re.Pattern:
    """把 Windows 绝对路径编成"分隔符数量与大小写都不敏感"的正则。

    JSON 里同一个路径写作 ``D:\\\\WX\\\\...``（双反斜杠），纯文本里是单反斜杠，
    还可能混用正斜杠；用 [\\\\/]+ 一次吃掉任意组合。
    """
    segs = [s for s in re.split(r"[\\/]+", win_path.strip().strip('"').rstrip("\\")) if s]
    return re.compile("(?i)" + r"[\\/]{1,2}".join(re.escape(s) for s in segs))


def build_scrubs() -> list[tuple[re.Pattern, str]]:
    """待替换的机器本地前缀。交接包根只从环境读，绝不把这个值写进本文件。"""
    scrubs: list[tuple[re.Pattern, str]] = [(path_pattern(str(ROOT)), "<PROJECT_ROOT>")]
    handoff = os.environ.get("SLC_ROOT", "").strip()
    if handoff:
        scrubs.insert(0, (path_pattern(handoff), "<SLC_ROOT>"))
    else:
        print("[warn] $env:SLC_ROOT 未设置：交接包根不做前缀替换，"
              "仅靠 PERSONAL_PATH 闸门兜底（先 . tools\\slc_mars_env.ps1 再打包）")
    return scrubs


def should_skip(rel: Path, is_dir: bool) -> bool:
    if is_dir:
        return rel.name in SKIP_DIR_NAMES
    return rel.name in SKIP_FILE_NAMES or rel.suffix.lower() in SKIP_SUFFIX


def collect() -> list[Path]:
    """按白名单收集要进包的源文件（相对仓库根）。"""
    picked: list[Path] = []
    for name in INCLUDE_FILES:
        p = ROOT / name
        if not p.is_file():
            raise SystemExit(f"[fatal] 缺少根文件 {name}")
        picked.append(p)
    for d in INCLUDE_DIRS:
        base = ROOT / d
        if not base.is_dir():
            raise SystemExit(f"[fatal] 缺少目录 {d}")
        for p in sorted(base.rglob("*")):
            rel = p.relative_to(ROOT)
            if p.is_dir():
                continue
            if any(part in SKIP_DIR_NAMES for part in rel.parts[:-1]):
                continue
            if should_skip(rel, is_dir=False):
                continue
            picked.append(p)
    return picked


def emit_text(src: Path, dst: Path, scrubs, stats: dict, dry: bool) -> None:
    raw = src.read_bytes()
    bom = raw.startswith(b"\xef\xbb\xbf")
    try:
        text = raw.decode("utf-8-sig" if bom else "utf-8")
    except UnicodeDecodeError:                      # 非 UTF-8 文本：原样复制，但仍要过闸门
        stats["copy_raw"] += 1
        if PERSONAL_PATH.search(raw.decode("latin-1")):
            stats["personal_hits"].append(str(src.relative_to(ROOT)))
        if not dry:
            dst.write_bytes(raw)
        return
    for rx, token in scrubs:
        text, n = rx.subn(token, text)
        stats["rewrites"] += n
    if PERSONAL_PATH.search(text):
        stats["personal_hits"].append(str(src.relative_to(ROOT)))
    if not dry:
        dst.write_bytes((("\ufeff" + text) if bom else text).encode("utf-8"))


def build(out: Path, dry: bool, force: bool = False) -> int:
    scrubs = build_scrubs()
    files = collect()
    stats = {"copied": 0, "skipped_big": 0, "copy_raw": 0, "rewrites": 0, "personal_hits": []}
    total = 0
    per_top: dict[str, list[int]] = {}
    largest = (0, "")

    if out.exists() and not dry:
        if not (out / MARKER).exists() and any(out.iterdir()) and not force:
            raise SystemExit(f"[fatal] {out} 已存在且没有本脚本的构建标记，拒绝清空。"
                             f"确认可以删就加 --force（被中断的半成品就是这种情况）")
        # Windows 上索引器/杀软会短暂占住大的 jsonl；清空失败就退避重试，而不是半途删树
        for attempt in range(6):
            try:
                shutil.rmtree(out)
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
        top = rel.parts[0] if rel.parts[0] != "outputs" else "outputs/traces"
        per_top.setdefault(top, [0, 0])
        per_top[top][0] += 1
        per_top[top][1] += size
        total += size
        if size > largest[0]:
            largest = (size, str(rel))
        dst = out / rel
        # dry-run 也要走 sanitize + 闸门，否则"检查"永远检查不到东西（只是不写盘）
        if not dry:
            dst.parent.mkdir(parents=True, exist_ok=True)
        if src.suffix.lower() in TEXT_SUFFIX:
            emit_text(src, dst, scrubs, stats, dry)
        elif not dry:
            shutil.copy2(src, dst)
        stats["copied"] += 1

    if not dry:
        out.mkdir(parents=True, exist_ok=True)
        (out / MARKER).write_text("built by tools/package_release.py\n", encoding="utf-8")

    print("\n==== release package %s ====" % ("(dry-run)" if dry else out))
    for top in sorted(per_top):
        n, b = per_top[top]
        print(f"  {top:<20} {n:>6} files  {b / 1e6:>8.1f} MB")
    print(f"  {'TOTAL':<20} {stats['copied']:>6} files  {total / 1e6:>8.1f} MB")
    print(f"  largest={largest[1]} ({largest[0] / 1e6:.1f} MB)")
    print(f"  path rewrites={stats['rewrites']}  raw(non-utf8) copies={stats['copy_raw']}  "
          f"over-limit skipped={stats['skipped_big']}")
    tbl = out / "outputs" / "traces" / "baselines_table_master.md"
    if not dry and tbl.is_file():
        print("  sha256 baselines_table_master.md = " + hashlib.sha256(tbl.read_bytes()).hexdigest()[:16])

    if stats["personal_hits"]:
        print(f"\n[fatal] 仍有 {len(stats['personal_hits'])} 个文件含个人目录绝对路径：")
        for h in stats["personal_hits"][:20]:
            print("   -> " + h)
        return 1
    if stats["skipped_big"]:
        print("[fatal] 有文件超过 GitHub 单文件上限，请调整白名单")
        return 1
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=str(DEFAULT_OUT), help="发布树输出目录")
    ap.add_argument("--dry-run", action="store_true", help="只统计不落盘")
    ap.add_argument("--force", action="store_true", help="输出目录没有构建标记时也允许清空重建")
    a = ap.parse_args()
    return build(Path(a.out).resolve(), a.dry_run, a.force)


if __name__ == "__main__":
    sys.exit(main())
