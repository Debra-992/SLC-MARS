# -*- coding: utf-8 -*-
"""Probe: one cheap LLM call to verify endpoint reachability from this shell. Read-only."""
import asyncio, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "knowledge_agents" / "baselines"))
from common import llm_settings


async def main():
    cfg = llm_settings()
    print("base_url:", (cfg["base_url"] or "")[:48], "model:", cfg["model"])
    if not cfg["base_url"] or not cfg["api_key"]:
        print("PROBE=NO_CONFIG")
        return
    from openai import AsyncOpenAI
    client = AsyncOpenAI(base_url=cfg["base_url"], api_key=cfg["api_key"],
                         timeout=60, max_retries=0)
    rsp = await client.chat.completions.create(
        model=cfg["model"],
        messages=[{"role": "user", "content": "reply with the single word: ok"}],
        max_tokens=4,
        extra_body={"enable_thinking": False},
    )
    print("PROBE=OK", rsp.choices[0].message.content.strip()[:20],
          "total_tokens=", getattr(rsp.usage, "total_tokens", None))


if __name__ == "__main__":
    asyncio.run(main())
