# -*- coding: utf-8 -*-
"""baseline runner 共享件：仅 stdlib，可被任意独立 venv（autogen/mdagents）直接拷贝路径引用。

公平性约定：
  * 同一题集：读 outputs/traces/baselines_input.json（由 export_inputs.py 从 v10 对比
    同一样本 manifest 导出，40 题 seed5，含候选基因）。
  * 同一指令：所有纯推理 baseline 用同一 _BASELINE_PROMPT（与原生 MetaGPT baseline 逐字一致）。
  * 同一端点：同一 LLM（qwen3.8-flash, openai 兼容模式），凭据由 SLC_LLM_* 环境变量
    提供（secrets.local.ps1 注入，仓库内 config2.yaml 只留占位符）。
  * 答案 schema 与管线 Answer 对齐：{qid,bucket,winner,answer,confidence,abstained,rationale}。
  * 失败诚实：LLM 不可达/解析失败 -> abstained=true，绝不伪造 winner。
"""
from __future__ import annotations

import json
import os
import re
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TRACES = ROOT / "outputs" / "traces"
INPUT_PATH = TRACES / "baselines_input.json"          # 无真值，可给任何 runner
GT_PATH = TRACES / "baselines_gt.json"                # 仅 merge 评分用，runner 禁读


def load_inputs() -> list[dict]:
    return json.loads(INPUT_PATH.read_text(encoding="utf-8"))


def answers_path(system: str) -> Path:
    return TRACES / f"answers_{system}.jsonl"


def load_done(system: str) -> dict[str, dict]:
    """resume：已落盘 qid -> 行（后写覆盖先写）。"""
    out: dict[str, dict] = {}
    p = answers_path(system)
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                try:
                    r = json.loads(line)
                    out[r["qid"]] = r
                except Exception:  # noqa BLE001
                    continue
    return out


def pending(system: str, inputs: list[dict]) -> list[dict]:
    """只跳过【成功】落盘的题；error 行（超时等瞬时故障）重跑，同 qid 后写覆盖先写。"""
    done = load_done(system)
    return [r for r in inputs
            if not (r["qid"] in done and not (done[r["qid"]].get("error") or ""))]


def append_answer(system: str, rec: dict) -> None:
    with answers_path(system).open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


# 与 knowledge_agents/baseline_metagpt._BASELINE_PROMPT 逐字一致（改一处需同步）
BASELINE_PROMPT = """你是生物医学问答助手。仅凭你自己的知识回答，不使用任何工具、检索或外部数据。
必须给出一个明确答案（这是原生 LLM 基线，不允许 abstain）：pairwise 必须二选一，membership 必须 true 或 false。

问题(桶={bucket}): {question}
{candidates_line}
请只输出 JSON（abstained 恒为 false）：
- pairwise: {{"winner": "<候选基因之一>", "confidence": 0~1, "abstained": false, "rationale": "..."}}
- membership: {{"answer": true/false, "confidence": 0~1, "abstained": false, "rationale": "..."}}
"""


def candidates_line(rec: dict) -> str:
    cands = rec.get("candidates") or []
    if rec.get("bucket") == "pairwise":
        return f"候选基因: {cands}"
    return f"目标基因: {cands[:1]}"


def build_prompt(rec: dict, context: str = "") -> str:
    base = BASELINE_PROMPT.format(bucket=rec["bucket"], question=rec["question"],
                                  candidates_line=candidates_line(rec))
    return (context + "\n\n" + base) if context else base


_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def extract_json(text: str) -> dict | None:
    m = _JSON_RE.search(text or "")
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def normalize(rec: dict, data: dict | None, *, system: str, error: str = "") -> dict:
    """LLM 原始 JSON -> 标准答案行。winner 不在候选内 / 缺字段 -> 记 abstain（诚实）。"""
    bucket = rec["bucket"]
    cands_u = {c.upper() for c in (rec.get("candidates") or [])}
    out = {"qid": rec["qid"], "bucket": bucket, "system": system,
           "winner": None, "answer": None, "confidence": 0.0,
           "abstained": True, "rationale": "", "error": error}
    if data:
        abst = bool(data.get("abstained"))
        w = data.get("winner") if bucket == "pairwise" else None
        a = data.get("answer") if bucket == "membership" else None
        if w and str(w).upper() not in cands_u:
            w = None
            abst = True
        if w is None and bucket == "pairwise":
            abst = True
        if a is None and bucket == "membership":
            abst = True
        try:
            conf = float(data.get("confidence", 0.0) or 0.0)
        except (TypeError, ValueError):
            conf = 0.0
        out.update(winner=w, answer=bool(a) if (bucket == "membership" and a is not None) else None,
                   confidence=conf, abstained=abst, rationale=str(data.get("rationale", ""))[:800])
    return out


# ---------------------------------------------------------------- LLM 端点（env 优先，openai 兼容）
# 凭据不在仓库里：config2.yaml 只有占位符，真值由 secrets.local.ps1 注入 SLC_LLM_* 环境变量。
_LLM_UNSET = "REPLACE_VIA_SECRETS_LOCAL_PS1"


def llm_settings() -> dict:
    """LLM 端点三键解析（被独立 venv 的 runner 共用）。

    优先级：环境变量 SLC_LLM_BASE_URL / SLC_LLM_API_KEY / SLC_LLM_MODEL
    （由 `secrets.local.ps1` 或 shell profile / CI secret store 导出）
    > config2.yaml 的 llm 块。yaml 里未被替换的占位符按"未配置"处理（返回空串），
    使调用方的 "未找到 llm.base_url/api_key" 分支给出可读报错，
    而不是把字面量当密钥发出去换一个莫名的 401。
    """
    cfg = ROOT / "config2.yaml"
    txt = cfg.read_text(encoding="utf-8", errors="replace") if cfg.exists() else ""
    out = {"base_url": "", "api_key": "", "model": ""}
    in_llm = False
    for line in txt.splitlines():
        if re.match(r"^llm:\s*$", line):
            in_llm = True
            continue
        if in_llm:
            if line and not line.startswith((" ", "\t")):
                break
            m = re.match(r"\s+(base_url|api_key|model):\s*(\S.*)$", line)
            if m:
                out[m.group(1)] = m.group(2).strip().strip("\"'")
    for k in out:
        envv = os.environ.get("SLC_LLM_" + k.upper(), "")
        if envv:
            out[k] = envv
        elif out[k] == _LLM_UNSET:
            out[k] = ""
    return out


# ---------------------------------------------------------------- token 计量（跳 baseline 统一）
def usage_triplet(rsp) -> tuple[int, int, int]:
    """从 openai 响应对象安全提取 (prompt, completion, total) token；缺失记 0。"""
    u = getattr(rsp, "usage", None)
    if u is None:
        return (0, 0, 0)
    p = int(getattr(u, "prompt_tokens", 0) or 0)
    c = int(getattr(u, "completion_tokens", 0) or 0)
    t = int(getattr(u, "total_tokens", 0) or (p + c))
    return (p, c, t)


class TokenSink:
    """进程级共享累计器（锁保护）。autogen/mdagents 池均单 worker 串行，
    一题开始 reset、结束 snapshot 即得该题用量；因 autogen 内部可能在**子线程**发
    create，故不能用 thread-local（否则跨线程 add 不可见→漏计）。"""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._v = [0, 0, 0, 0]   # prompt, completion, total, calls

    def reset(self) -> None:
        with self._lock:
            self._v = [0, 0, 0, 0]

    def add(self, p: int, c: int, t: int) -> None:
        with self._lock:
            self._v[0] += p; self._v[1] += c; self._v[2] += t; self._v[3] += 1

    def snapshot(self) -> dict:
        with self._lock:
            v = list(self._v)
        return {"prompt": v[0], "completion": v[1], "total": v[2], "calls": v[3]}


def install_openai_token_tracking(sink: TokenSink) -> bool:
    """monkeypatch 同步 openai 的 Chat Completions.create，把每次调用的 usage 累进 sink。
    返回是否安装成功（版本路径不符则 False，调用方据此降级）。不改返回值/异常语义。"""
    try:
        from openai.resources.chat.completions import Completions
    except Exception:  # noqa BLE001
        return False
    real = Completions.create
    if getattr(real, "__tok_wrapped__", False):
        return True

    def create(self, *a, **kw):
        rsp = real(self, *a, **kw)
        try:
            if not kw.get("stream"):
                p, c, t = usage_triplet(rsp)
                sink.add(p, c, t)
        except Exception:  # noqa BLE001 —— 计量绝不干扰作答
            pass
        return rsp

    create.__tok_wrapped__ = True
    Completions.create = create
    return True


def install_metagpt_token_tracking(sink: TokenSink) -> bool:
    """把 MetaGPT 的 CostManager.update_cost 镜像进 sink（每个 LLM 请求调一次）。
    MetaGPT provider 走 AsyncOpenAI + stream=True，openai 层难直接抓 usage，但其
    provider 总会调 update_cost 记账 → 在此拦截最稳。内部才 import metagpt（不污染无
    metagpt 的 venv）。返回是否安装成功。"""
    try:
        from metagpt.utils.cost_manager import CostManager
    except Exception:  # noqa BLE001
        return False
    real = CostManager.update_cost
    if getattr(real, "__tok_wrapped_mg__", False):
        return True

    def update_cost(self, prompt_tokens, completion_tokens, model):
        try:
            p = int(prompt_tokens or 0); c = int(completion_tokens or 0)
            if p + c > 0:
                sink.add(p, c, p + c)
        except Exception:  # noqa BLE001 —— 计量绝不干扰作答
            pass
        return real(self, prompt_tokens, completion_tokens, model)

    update_cost.__tok_wrapped_mg__ = True
    CostManager.update_cost = update_cost
    return True


def install_temperature_zero() -> bool:
    """env SLC_FORCE_TEMP0=1 时，强制本进程所有 openai chat.completions.create 的
    temperature=0（同步 + 异步均盖）。管线的 LLM 全走 AsyncOpenAI.chat.completions.create
    （prior/debate/decision/L2）且 MetaGPT provider 也走 AsyncOpenAI → 一个中央拦截点即可
    锁死全部采样，消除 LLM 非确定性对 R2 开关 A/B 配对的污染。未设 env 则不做任何事（零副作用）。
    仅改 temperature kwarg，不动返回值/异常语义。"""
    if os.environ.get("SLC_FORCE_TEMP0", "0").lower() not in {"1", "true", "on", "yes"}:
        return False
    try:
        from openai.resources.chat.completions import Completions, AsyncCompletions
    except Exception:  # noqa BLE001
        return False
    real_s = Completions.create
    if not getattr(real_s, "__temp0__", False):
        def create(self, *a, **kw):
            kw["temperature"] = 0
            return real_s(self, *a, **kw)
        create.__temp0__ = True
        Completions.create = create
    real_a = AsyncCompletions.create
    if not getattr(real_a, "__temp0__", False):
        async def acreate(self, *a, **kw):
            kw["temperature"] = 0
            return await real_a(self, *a, **kw)
        acreate.__temp0__ = True
        AsyncCompletions.create = acreate
    return True
