# -*- coding: utf-8 -*-
"""离线守卫：仓库里任何被跟踪的源码/配置/文档都不得再出现凭据字面量。

背景：LLM api_key、Neo4j 口令、带鉴权信息的 MONGO_URI 曾明文躺在
`run_route_b.ps1` / `tools/slc_mars_env.ps1` / `config2.yaml` 里，2026-11-03 迁到
环境变量 + 未纳入版本管理的 `secrets.local.ps1`。本测试防止再次被写回来。

只扫"人会提交"的文件：数据与结果产物（outputs/、data/、evaluation/）里有
`sk-literature-v4-opt-in` 这类含连字符的内部标签，属正常内容，不是密钥。

Run:  .venv/Scripts/python.exe tools/test_no_secrets.py
"""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# 第三方 / 生成物 / 数据：不扫（第三方自带示例密钥，数据里是内部标签与题干）
SKIP_DIRS = {".venv", ".cache", ".git", "MetaGPT", "outputs", "data", "evaluation",
             "node_modules", "__pycache__", "R-libs", "autogen-venv", "mdagents-venv",
             "mcp-venv", "MDAgents", "slc-side-cache", "slc-tools-cache"}
SCAN_SUFFIX = {".py", ".ps1", ".psm1", ".yaml", ".yml", ".json", ".md", ".tex", ".txt",
               ".sh", ".cfg", ".ini", ".toml", ".R", ".html"}

# 占位符白名单：模板/example 文件里本来就该长这样
PLACEHOLDER = re.compile(r"REPLACE|COME_FROM|_FROM_ENV|YOUR-|xxx|XXX|<[^>]+>|\$\{", re.I)

RULES = [
    # api_key 形态：sk- 后 20+ 位纯字母数字（内部标签 sk-literature-v4-opt-in 有连字符，不会误报）
    ("openai-style key", re.compile(r"\bsk-[A-Za-z0-9]{20,}")),
    # 带口令的连接串：mongodb://user:PASS@ / bolt+... 等
    ("password in connection URI",
     re.compile(r"\b(?:mongodb|mongodb\+srv|mysql|postgres(?:ql)?|bolt(?:\+s)?|neo4j)"
                r"[a-z+]*://[^:/\s\"']+:[^@\s\"']{8,}@", re.I)),
    # 显式赋值口令
    ("literal password assignment",
     re.compile(r"(?:api_?key|secret|passwd|password|token)\s*(?:=|:)\s*[\"']([^\"']{8,})[\"']",
                re.I)),
    # Bearer 硬编码
    ("bearer token", re.compile(r"Bearer\s+([A-Za-z0-9._\-]{20,})")),
]


def looks_secret(fragment):
    """把 `api_key=cfg["api_key"]` 这类"引用/字段名"与真密钥分开：真密钥是高熵的。"""
    v = fragment.strip().strip("\"'")
    if not v or PLACEHOLDER.search(v):
        return False
    if re.fullmatch(r"[a-z][a-z0-9_]*", v) and len(v) < 16:
        return False                                      # 字段名/小写短词，如 api_key、postgres
    if " " in v:                                        # 散文/示例说明
        return False
    if not any(c.isdigit() for c in v) and len(v) < 16:  # 低熵短词，如 postgres
        return False
    return True


def offenders():
    out = []
    for p in sorted(ROOT.rglob("*")):
        if not p.is_file() or p.suffix.lower() not in SCAN_SUFFIX:
            continue
        if any(part in SKIP_DIRS for part in p.relative_to(ROOT).parts):
            continue
        # 唯一允许持有真值的文件：本机 secrets.local.ps1（.gitignore 已挡住，见下方检查）
        if p.name == "secrets.local.ps1":
            continue
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for i, line in enumerate(text.splitlines(), 1):
            for label, rx in RULES:
                m = rx.search(line)
                if not m:
                    continue
                frag = m.group(1) if m.groups() else m.group(0)
                if looks_secret(frag):
                    out.append((f"{p.relative_to(ROOT)}:{i}", label, line.strip()[:90]))
    return out


def gitignore_covers_vault():
    gi = ROOT / ".gitignore"
    if not gi.exists():
        return False, ".gitignore 不存在"
    names = {ln.strip() for ln in gi.read_text(encoding="utf-8", errors="replace").splitlines()}
    for pat in ("secrets.local.ps1", "*.local.ps1"):
        if pat in names:
            return True, pat
    return False, "secrets.local.ps1 未被 .gitignore 覆盖"


ok = True


def check(name, cond, detail=""):
    global ok
    ok = ok and bool(cond)
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))


bad = offenders()
check("no credential literal in tracked source/config/docs", not bad,
      f"({len(bad)} hit(s))" if bad else "(scanned source, configs, docs)")
for where, label, text in bad[:12]:
    print(f"   -> {where} [{label}] {text}")
    if len(bad) > 12:
        print(f"   ... 另有 {len(bad) - 12} 处")
        break

covered, pat = gitignore_covers_vault()
check("local vault is git-ignored", covered, f"({pat})")
check("template exists for the vault", (ROOT / "secrets.example.ps1").exists())
check("repo config2.yaml carries no real key",
      not re.search(r"\bsk-[A-Za-z0-9]{20,}",
                    (ROOT / "config2.yaml").read_text(encoding="utf-8", errors="replace")))

print("\n==== " + ("ALL PASS" if ok else "SOME FAILED") + " ====")
raise SystemExit(0 if ok else 1)
