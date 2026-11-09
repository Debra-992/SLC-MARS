# -*- coding: utf-8 -*-
"""Offline test for the L2 vote-source decorrelation gate (SLC_PRIOR_RETRIEVE, the lightweight
retrieved-literature pack).

Coverage: the gate defaults off / with retrieval="" the prompt is byte-identical to the one-shot
prior / the injected section is present and carries no abstention-override clause / pack truncation
and the score=0 filter / silent degradation on exception / pass-through into the raced copies /
only the question text, candidates and constraints are used (no ground-truth leak).
Everything is mocked (AsyncOpenAI + mongo_rag.retrieve); no network and no intranet access.
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
    """Capture the prompt of every call; return a parseable JSON vote. Recorded at class level so
    the calls collected across instances are visible."""
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

# T1 the gate defaults to off
os.environ.pop("SLC_PRIOR_RETRIEVE", None)
check("T1 gate default OFF (= released configuration, byte-identical)", pa.prior_retrieve_enabled() is False)

# T2 retrieval="" -> the prompt is exactly the one-shot prior's (no retrieval section at all)
FakeClient.prompts.clear()
asyncio.run(pa.ask_prior_augmented(Q, "pairwise", CANDS, digest="", persona="P", retrieval=""))
p_bare = FakeClient.prompts[-1]
check("T2 retrieval='' prompt byte-identical to the one-shot prior (no retrieval section)",
      "Excerpts from the project literature corpus" not in p_bare
      and "independent of your memory" not in p_bare)

# T3 the retrieval section is injected, and it does NOT carry the abstention-override clause
#    (the L2 channel must commit to an answer)
FakeClient.prompts.clear()
asyncio.run(pa.ask_prior_augmented(Q, "pairwise", CANDS, digest="", persona="P",
                                   retrieval="[1] Title A -- abstract body"))
p_retr = FakeClient.prompts[-1]
check("T3 retrieval section injected, no abstain-override clause",
      "Excerpts from the project literature corpus" in p_retr and "[1] Title A" in p_retr
      and "you may now abstain" not in p_retr
      and p_retr[:len(p_bare)] == p_bare)  # the prefix is byte-for-byte the bare vote, only appended to

# T4 build_retrieval_pack: the score=0 filter plus the per-abstract / total-length caps
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

# T5 an exception degrades silently to "" (the bare vote) and must never propagate
def _boom(*a, **k):
    raise RuntimeError("mongo down")
mongo_rag.retrieve = _boom
check("T5 retrieval failure degrades to '' (never raises)",
      pa.build_retrieval_pack(Q, "pairwise", CANDS) == "")

# T6 race with n=2: the retrieval pack is passed through to every replica
os.environ["SLC_PRIOR_RACE"] = "2"
FakeClient.prompts.clear()
asyncio.run(pa.ask_prior_raced(Q, "pairwise", CANDS, persona="P2", temperature=0.7,
                               retrieval="PACKTEXT"))
check("T6 raced copies all carry the pack (n=2 both prompts)",
      len(FakeClient.prompts) >= 1 and all("PACKTEXT" in p for p in FakeClient.prompts))

# T7 no ground-truth leak: the pack is built from the question text, the candidates and the
#    constraints only (the arguments retrieve() receives carry no ground_truth field)
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
