# -*- coding: utf-8 -*-
"""离线单测：LLM 端点三键的解析优先级（baselines.common.llm_settings）。

凭据迁移（2026-11-03）后的约定：
  1. `SLC_LLM_BASE_URL/API_KEY/MODEL` 环境变量优先；
  2. 其次仓库 `config2.yaml` 的 llm 块；
  3. yaml 里未被替换的占位符 == 没配（返回空串），绝不能把占位符字面量当密钥发出去；
  4. 环境变量存在但为空串，视作"没提供"，不得把 yaml 里的值洗成空。
第 3 条最容易被无声破坏：占位符一旦漏出去，症状是远端 401，而不是本地一句"未配置"。

测试里的假密钥一律写成 XXXXXXXX 掩码：tools/test_no_secrets.py 会扫全仓库源码，
密钥形状的夹具（哪怕明显是编的）会被它判成凭据泄漏。

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
# 进程进入时的状态：是否在 dot-source 过 slc_mars_env.ps1 的 shell 里跑，都不影响结论，
# 只要求 scope 退出后回到这个状态。
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
    """把 config2.yaml 换成给定内容、把 SLC_LLM_* 换成给定值；块内调用 llm_settings()。

    ROOT 与环境变量都在块内保持生效（否则后续调用会读回真实仓库配置，断言就失去意义），
    退出时无条件复原，保证测试自身不污染进程环境。
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


# --- 1) 未替换的占位符 = 未配置，且绝不外发字面量 --------------------------------
with scope(PLACEHOLDER_YAML):
    cfg = common.llm_settings()
    check("placeholders resolve to unconfigured (never sent as a key)",
          all(v == "" for v in cfg.values()), f"-> {cfg}")

# --- 2) env 缺席时回退 yaml -----------------------------------------------------
with scope(REAL_YAML):
    cfg = common.llm_settings()
    check("yaml fallback honoured when env absent",
          cfg == {"base_url": "https://yaml.example/v1",
                  "api_key": "yaml-key-XXXXXXXX", "model": "yaml-model"}, f"-> {cfg}")

# --- 3) 单个键由 env 覆盖，其余仍取 yaml ----------------------------------------
with scope(REAL_YAML, SLC_LLM_API_KEY="env-key-XXXXXXXX"):
    cfg = common.llm_settings()
    check("one env key overrides only that key",
          cfg["api_key"] == "env-key-XXXXXXXX"
          and cfg["base_url"] == "https://yaml.example/v1"
          and cfg["model"] == "yaml-model", f"-> {cfg}")

# --- 4) 三键全由 env 提供 --------------------------------------------------------
with scope(REAL_YAML, SLC_LLM_BASE_URL="https://env.example/v1",
           SLC_LLM_API_KEY="env-key-XXXXXXXX", SLC_LLM_MODEL="env-model"):
    cfg = common.llm_settings()
    check("env wins over yaml for all three keys",
          cfg == {"base_url": "https://env.example/v1",
                  "api_key": "env-key-XXXXXXXX", "model": "env-model"}, f"-> {cfg}")

# --- 5) 空串环境变量视作未提供 ---------------------------------------------------
with scope(REAL_YAML, SLC_LLM_API_KEY=""):
    cfg = common.llm_settings()
    check("empty env var does not blank out the yaml value",
          cfg["api_key"] == "yaml-key-XXXXXXXX", f"-> {cfg['api_key']}")

# --- 6) 退出 scope 后进程状态与进入时一致（测试不污染自身） ---------------------
check("env restored to process-start state after scopes",
      all(os.environ.get(k) == _START_ENV[k] for k in KEYS),
      f"(ambient: {'set' if _START_ENV['SLC_LLM_API_KEY'] else 'absent'})")
check("common.ROOT restored to the repo", common.ROOT == _START_ROOT, f"-> {common.ROOT}")

print("\n==== " + ("ALL PASS" if ok else "SOME FAILED") + " ====")
raise SystemExit(0 if ok else 1)
