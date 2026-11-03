"""查询 API 返回的实际 model 名称（含快照版本）。凭据走 SLC_LLM_* 环境变量。"""
import asyncio, sys
from pathlib import Path

from openai import AsyncOpenAI

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from knowledge_agents.baselines.common import llm_settings  # noqa: E402

cfg = llm_settings()
if not cfg["base_url"] or not cfg["api_key"]:
    raise SystemExit("LLM 端点未配置：先 dot-source tools\\slc_mars_env.ps1"
                     "（它加载仓库根未纳入版本管理的 secrets.local.ps1）。")

client = AsyncOpenAI(base_url=cfg["base_url"], api_key=cfg["api_key"],
                     timeout=15, max_retries=0)

async def main():
    rsp = await client.chat.completions.create(
        model=cfg["model"],
        messages=[{"role": "user", "content": "Say only: OK"}],
        temperature=0,
        max_tokens=5,
    )
    print(f"request model : {cfg['model']}")
    print(f"response model: {rsp.model}")
    print(f"response id   : {rsp.id}")
    print(f"usage         : {rsp.usage}")
    print(f"content       : {rsp.choices[0].message.content}")

asyncio.run(main())