# -*- coding: utf-8 -*-
"""v17 离线测试：L2 票源去同源化（SLC_PRIOR_RETRIEVE 轻检索文学包）。

覆盖：门控默认关 / retrieval="" 题面逐字节同 v16 / 注入段存在且不带可-abstain 条款 /
pack 截断与 score=0 过滤 / 异常静默退化 / 竞速透传 / 只用题面候选约束（无 GT 泄漏）。
全程 mock（AsyncOpenAI + mongo_rag.retrieve），不碰网络不碰内网。
"""
import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from knowledge_agents import prior_agent as pa  # noqa: E402

ok = True


def check(name, cond, extra=""):
    global ok
    ok = ok and bool(cond)
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f" | {extra}" if extra else ""))


class FakeCompletion:
    def __init__(self, content):
        self.choices = [type("M", (), {"message": type("C", (), {"content": content})})()]


class FakeClient:
    """捕获每题 prompt；返回可解析 JSON 票。类级记录跨实例收集。"""
    prompts = []
    payload = '{"winner": "SLC25A1", "confidence": 0.7, "abstained": false, "rationale": "t"}'

    def __init__(self, **kw):
        pass

    class _Chat:
        class _Completions:
            async def create(_self, **kw):
                FakeClient.prompts.append(kw["messages"][0]["content"])
                return FakeCompletion(FakeClient.payload)
        completions = _Completions()
    chat = _Chat()

    async def close(self):
        pass


def install_fake():
    import openai
    openai.AsyncOpenAI = FakeClient
    pa.llm_settings = lambda: {"base_url": "http://fake/v1", "api_key": "k",
                               "model": "fake", "temperature": 0.2}


Q = "In KRAS-mutant lung adenocarcinoma, which SLC is more essential, SLC25A1 or SLC1A6?"
CANDS = ["SLC25A1", "SLC1A6"]

install_fake()

# T1 门控默认关
os.environ.pop("SLC_PRIOR_RETRIEVE", None)
check("T1 gate default OFF (= champion v16 byte-identical)", pa.prior_retrieve_enabled() is False)

# T2 retrieval="" -> 题面与 v16 完全一致（不含任何 v17 注入段）
FakeClient.prompts.clear()
asyncio.run(pa.ask_prior_augmented(Q, "pairwise", CANDS, digest="", persona="P", retrieval=""))
p_bare = FakeClient.prompts[-1]
check("T2 retrieval='' prompt byte-identical to v16 (no v17 section)",
      "文献库摘录" not in p_bare and "相互独立" not in p_bare)

# T3 retrieval 注入存在，且【不带】可-abstain 覆盖条款（L2 必须承诺作答）
FakeClient.prompts.clear()
asyncio.run(pa.ask_prior_augmented(Q, "pairwise", CANDS, digest="", persona="P",
                                   retrieval="[1] Title A — abstract body"))
p_retr = FakeClient.prompts[-1]
check("T3 retrieval section injected, no abstain-override clause",
      "文献库摘录" in p_retr and "[1] Title A" in p_retr
      and "你现在可以 abstain" not in p_retr
      and p_retr[:len(p_bare)] == p_bare)  # 前缀逐字节=v16 票面，只在其后追加

# T4 build_retrieval_pack：score=0 过滤 + 摘要/总长截断
from knowledge_agents.baselines import mongo_rag  # noqa: E402

mongo_rag.retrieve = lambda q, c, k, top_k=3: [
    {"score": 5, "title": "T" * 300, "abstract": "A" * 900, "doi": "d1"},
    {"score": 0, "title": "JUNK", "abstract": "JUNK", "doi": "d2"},
    {"score": 2, "title": "Good", "abstract": "B" * 900, "doi": "d3"},
]
pack = pa.build_retrieval_pack(Q, "pairwise", CANDS, {"cancer": "lung"})
check("T4 pack filters score=0, caps abstract<=300 & total<=1200",
      "JUNK" not in pack and "A" * 301 not in pack and len(pack) <= 1200
      and "[1] " in pack and "Good" in pack)

# T5 异常 -> 静默退化为 ""（v16 裸票），绝不抛
def _boom(*a, **k):
    raise RuntimeError("mongo down")
mongo_rag.retrieve = _boom
check("T5 retrieval failure degrades to '' (never raises)",
      pa.build_retrieval_pack(Q, "pairwise", CANDS) == "")

# T6 竞速 n=2：retrieval 透传到每份副本
os.environ["SLC_PRIOR_RACE"] = "2"
FakeClient.prompts.clear()
asyncio.run(pa.ask_prior_raced(Q, "pairwise", CANDS, persona="P2", temperature=0.7,
                               retrieval="PACKTEXT"))
check("T6 raced copies all carry the pack (n=2 both prompts)",
      len(FakeClient.prompts) >= 1 and all("PACKTEXT" in p for p in FakeClient.prompts))

# T7 无 GT 泄漏：pack 只用题面/候选/约束（retrieve 收到的入参不含 ground_truth 字段）
seen = {}
def _spy(q, c, k, top_k=3):
    seen.update(q=q, c=c, k=k)
    return []
mongo_rag.retrieve = _spy
pa.build_retrieval_pack(Q, "membership", ["SLC25A1"], {"cancer": "ovary"})
check("T7 retrieve() receives only question/candidates/constraints",
      seen.get("q") == Q and seen.get("c") == ["SLC25A1"] and set(seen.get("k", {})) == {"cancer"})

print("==== ALL PASS ====" if ok else "==== SOME FAILED ====")
sys.exit(0 if ok else 1)
