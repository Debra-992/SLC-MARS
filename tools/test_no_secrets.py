# -*- coding: utf-8 -*-
"""Offline guard: no tracked source, configuration or document in the repository may carry a
credential literal.

Background: the LLM api_key, the Neo4j password and the MONGO_URI with its embedded authentication
once sat in plain text in `run_route_b.ps1`, `tools/slc_mars_env.ps1` and `config2.yaml`. They now
live in environment variables plus the untracked `secrets.local.ps1`, and this test keeps them from
being written back in.

Only files a person would commit are scanned. Data and result artifacts (outputs/, data/,
evaluation/) contain internal hyphenated labels such as `sk-literature-v4-opt-in`, which are ordinary
content and not keys.

Run:  .venv/Scripts/python.exe tools/test_no_secrets.py
"""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Third-party code, generated files and data are not scanned (upstreams ship example keys, and the
# data carries internal labels and question stems)
SKIP_DIRS = {".venv", ".cache", ".git", "MetaGPT", "outputs", "data", "evaluation",
             "node_modules", "__pycache__", "R-libs", "autogen-venv", "mdagents-venv",
             "mcp-venv", "MDAgents", "slc-side-cache", "slc-tools-cache"}
SCAN_SUFFIX = {".py", ".ps1", ".psm1", ".yaml", ".yml", ".json", ".md", ".tex", ".txt",
               ".sh", ".cfg", ".ini", ".toml", ".R", ".html"}

# Placeholder whitelist: a template / example file is supposed to look exactly like this
PLACEHOLDER = re.compile(r"REPLACE|COME_FROM|_FROM_ENV|YOUR-|xxx|XXX|<[^>]+>|\$\{", re.I)

RULES = [
    # api_key shape: sk- followed by 20+ alphanumeric characters (the internal label
    # sk-literature-v4-opt-in contains hyphens, so it is not reported)
    ("openai-style key", re.compile(r"\bsk-[A-Za-z0-9]{20,}")),
    # a connection URI that embeds a password: mongodb://user:PASS@, bolt+... and friends
    ("password in connection URI",
     re.compile(r"\b(?:mongodb|mongodb\+srv|mysql|postgres(?:ql)?|bolt(?:\+s)?|neo4j)"
                r"[a-z+]*://[^:/\s\"']+:[^@\s\"']{8,}@", re.I)),
    # an explicitly assigned password
    ("literal password assignment",
     re.compile(r"(?:api_?key|secret|passwd|password|token)\s*(?:=|:)\s*[\"']([^\"']{8,})[\"']",
                re.I)),
    # a hardcoded Bearer token
    ("bearer token", re.compile(r"Bearer\s+([A-Za-z0-9._\-]{20,})")),
]


def looks_secret(fragment):
    """Separate a reference or field name such as `api_key=cfg["api_key"]` from a real key: a real
    key is high-entropy."""
    v = fragment.strip().strip("\"'")
    if not v or PLACEHOLDER.search(v):
        return False
    if re.fullmatch(r"[a-z][a-z0-9_]*", v) and len(v) < 16:
        return False                                      # a field name or a short lowercase word, e.g. api_key, postgres
    if " " in v:                                        # prose or an example note
        return False
    if not any(c.isdigit() for c in v) and len(v) < 16:  # a low-entropy short word, e.g. postgres
        return False
    return True


def offenders():
    out = []
    for p in sorted(ROOT.rglob("*")):
        if not p.is_file() or p.suffix.lower() not in SCAN_SUFFIX:
            continue
        if any(part in SKIP_DIRS for part in p.relative_to(ROOT).parts):
            continue
        # the one file allowed to hold real values: the local secrets.local.ps1 (already blocked by
        # .gitignore, which the check below verifies)
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
        return False, ".gitignore does not exist"
    names = {ln.strip() for ln in gi.read_text(encoding="utf-8", errors="replace").splitlines()}
    for pat in ("secrets.local.ps1", "*.local.ps1"):
        if pat in names:
            return True, pat
    return False, "secrets.local.ps1 is not covered by .gitignore"


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
    print(f"   ... and {len(bad) - 12} more")

covered, pat = gitignore_covers_vault()
check("local vault is git-ignored", covered, f"({pat})")
check("template exists for the vault", (ROOT / "secrets.example.ps1").exists())
check("repo config2.yaml carries no real key",
      not re.search(r"\bsk-[A-Za-z0-9]{20,}",
                    (ROOT / "config2.yaml").read_text(encoding="utf-8", errors="replace")))

print("\n==== " + ("ALL PASS" if ok else "SOME FAILED") + " ====")
raise SystemExit(0 if ok else 1)
