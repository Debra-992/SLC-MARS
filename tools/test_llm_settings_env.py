# -*- coding: utf-8 -*-
"""Offline unit test for the resolution order of the three LLM endpoint keys
(baselines.common.llm_settings).

The credential contract:
  1. the `SLC_LLM_BASE_URL / SLC_LLM_API_KEY / SLC_LLM_MODEL` environment variables win;
  2. then the `llm` block of the repository `config2.yaml`;
  3. a placeholder that was never substituted in the yaml means "not configured" (an empty string is
     returned) -- a placeholder literal must never be sent out in place of a key;
  4. an environment variable that exists but is empty counts as "not supplied" and must not blank
     out the value taken from the yaml.
Rule 3 is the easiest one to break silently: once a placeholder leaks out, the symptom is a 401 from
the remote endpoint rather than a clear local "not configured" message.

Fake keys in this test are always written as an XXXXXXXX mask: tools/test_no_secrets.py scans every
source file in the repository, and any key-shaped fixture (however obviously invented) is reported by
it as a credential leak.

Run:  .venv/Scripts/python.exe tools/test_llm_settings_env.py
"""
import os
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from knowledge_agents.baselines import common  # noqa: E402

KEYS = ("SLC_LLM_BASE_URL", "SLC_LLM_API_KEY", "SLC_LLM_MODEL")
# The state at process entry: it does not matter whether this runs inside a shell that has
# dot-sourced slc_mars_env.ps1, only that the state is restored once every scope below has exited.
_START_ENV = {k: os.environ.get(k) for k in KEYS}
_START_ROOT = common.ROOT
PLACEHOLDER_YAML = (
    "llm:\n"
    "  api_type: openai\n"
    "  base_url: REPLACE_VIA_SECRETS_LOCAL_PS1\n"
    "  api_key: REPLACE_VIA_SECRETS_LOCAL_PS1\n"
    "  model: REPLACE_VIA_SECRETS_LOCAL_PS1\n"
)
REAL_YAML = (
    "llm:\n"
    "  api_type: openai\n"
    "  base_url: https://yaml.example/v1\n"
    '  api_key: "yaml-key-XXXXXXXX"\n'
    "  model: yaml-model\n"
)

ok = True


def check(name, cond, detail=""):
    global ok
    ok = ok and bool(cond)
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))


@contextmanager
def scope(yaml_body, **env):
    """Replace config2.yaml with the given body and SLC_LLM_* with the given values, then call
    llm_settings() inside the block.

    ROOT and the environment variables both stay in force for the whole block (a later call would
    otherwise read the real repository config back and the assertions would mean nothing), and both
    are restored unconditionally on exit so the test cannot pollute the process environment.
    """
    saved_root, saved_env = common.ROOT, {k: os.environ.get(k) for k in KEYS}
    with tempfile.TemporaryDirectory() as td:
        (Path(td) / "config2.yaml").write_text(yaml_body, encoding="utf-8")
        common.ROOT = Path(td)
        for k in KEYS:
            os.environ.pop(k, None)
        for k, v in env.items():
            if v is not None:
                os.environ[k] = v
        try:
            yield
        finally:
            common.ROOT = saved_root
            for k, v in saved_env.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v


# --- 1) unsubstituted placeholders mean unconfigured, and are never sent out as a key ----------
with scope(PLACEHOLDER_YAML):
    cfg = common.llm_settings()
    check("placeholders resolve to unconfigured (never sent as a key)",
          all(v == "" for v in cfg.values()), f"-> {cfg}")

# --- 2) fall back to the yaml when the environment is absent -----------------------------------
with scope(REAL_YAML):
    cfg = common.llm_settings()
    check("yaml fallback honoured when env absent",
          cfg == {"base_url": "https://yaml.example/v1",
                  "api_key": "yaml-key-XXXXXXXX", "model": "yaml-model"}, f"-> {cfg}")

# --- 3) one environment key overrides that key only, the rest still come from the yaml ---------
with scope(REAL_YAML, SLC_LLM_API_KEY="env-key-XXXXXXXX"):
    cfg = common.llm_settings()
    check("one env key overrides only that key",
          cfg["api_key"] == "env-key-XXXXXXXX"
          and cfg["base_url"] == "https://yaml.example/v1"
          and cfg["model"] == "yaml-model", f"-> {cfg}")

# --- 4) all three keys supplied by the environment ----------------------------------------------
with scope(REAL_YAML, SLC_LLM_BASE_URL="https://env.example/v1",
           SLC_LLM_API_KEY="env-key-XXXXXXXX", SLC_LLM_MODEL="env-model"):
    cfg = common.llm_settings()
    check("env wins over yaml for all three keys",
          cfg == {"base_url": "https://env.example/v1",
                  "api_key": "env-key-XXXXXXXX", "model": "env-model"}, f"-> {cfg}")

# --- 5) an empty environment variable counts as not supplied -----------------------------------
with scope(REAL_YAML, SLC_LLM_API_KEY=""):
    cfg = common.llm_settings()
    check("empty env var does not blank out the yaml value",
          cfg["api_key"] == "yaml-key-XXXXXXXX", f"-> {cfg['api_key']}")

# --- 6) after every scope the process state matches what it was at entry (no self-pollution) ---
check("env restored to process-start state after scopes",
      all(os.environ.get(k) == _START_ENV[k] for k in KEYS),
      f"(ambient: {'set' if _START_ENV['SLC_LLM_API_KEY'] else 'absent'})")
check("common.ROOT restored to the repo", common.ROOT == _START_ROOT, f"-> {common.ROOT}")

print("\n==== " + ("ALL PASS" if ok else "SOME FAILED") + " ====")
raise SystemExit(0 if ok else 1)
